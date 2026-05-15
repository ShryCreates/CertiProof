"""
CertiProof Backend — AI Certificate Verification System
FastAPI + SQLite + JWT Auth + Real ML Pipeline
"""

import os
import sys
import uuid
import time
import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Text, Integer, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship, DeclarativeBase
import bcrypt as _bcrypt
from jose import JWTError, jwt

# ── Add ml/ to path so we can import forgery_detector & field_extractor ──────
ML_DIR = Path(__file__).parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from forgery_detector import build_model, predict, MODEL_CONFIG
from field_extractor import FieldExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("certiproof")

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    logger.info("Loaded .env configuration")
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# ============ CONFIG ============
VERSION    = "2.5.0"
SECRET_KEY = os.environ.get("SECRET_KEY", "certiproof-secret-key-change-in-production")
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# CORS — comma-separated origins from env, fallback to localhost dev ports
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8080,http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

UPLOAD_DIR  = Path("./uploads")
HEATMAP_DIR = Path("./heatmaps")
REPORT_DIR  = Path("./reports")
CHECKPOINT  = ML_DIR / "checkpoints" / "best_model.pth"

for d in [UPLOAD_DIR, HEATMAP_DIR, REPORT_DIR]:
    d.mkdir(exist_ok=True)

# ── Gmail SMTP config (set in .env) ──────────────────────────────────────────
SMTP_HOST     = os.environ.get("SMTP_HOST",     "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER     = os.environ.get("SMTP_USER",     "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
EMAIL_ENABLED = bool(SMTP_USER and SMTP_PASSWORD)
logger.info("Email enabled: %s | SMTP_USER: %s", EMAIL_ENABLED, SMTP_USER or "NOT SET")

# ============ LOAD ML MODELS (lazy — loaded once on first startup event) ============
_device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_forgery_model = None
_field_extractor = None

def _load_models():
    global _forgery_model, _field_extractor
    if _forgery_model is not None:
        return  # already loaded
    logger.info("Loading forgery detection model…")
    model = build_model().to(_device)
    if CHECKPOINT.exists():
        ckpt = torch.load(CHECKPOINT, map_location=_device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint (epoch %d, val_auc=%.4f)",
                    ckpt.get("epoch", 0), ckpt.get("val_auc", 0))
    else:
        logger.warning("No checkpoint at %s — using random weights", CHECKPOINT)
    model.eval()
    _forgery_model   = model
    _field_extractor = FieldExtractor(use_layoutlmv3=False)
    logger.info("ML models ready on device: %s", _device)

# ============ DATABASE ============
DATABASE_URL = "sqlite:///./certiproof.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

pwd_context = None  # replaced by direct bcrypt calls below
security = HTTPBearer()


class User(Base):
    __tablename__ = "users"
    id              = Column(String,   primary_key=True, default=lambda: str(uuid.uuid4()))
    email           = Column(String,   unique=True, index=True, nullable=False)
    full_name       = Column(String,   nullable=False)
    hashed_password = Column(String,   nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow)
    verifications   = relationship("Verification", back_populates="user")


class Verification(Base):
    __tablename__ = "verifications"
    id                = Column(String,  primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id           = Column(String,  ForeignKey("users.id"), nullable=False)
    filename          = Column(String,  nullable=False)
    verdict           = Column(String,  nullable=False)
    trust_score       = Column(Integer, nullable=False)
    forgery_score     = Column(Float,   nullable=False)
    field_confidence  = Column(Float,   nullable=False)
    nlp_anomaly_score = Column(Float,   nullable=False)
    institution_match = Column(Boolean, nullable=False)
    institution_name  = Column(String,  nullable=True)
    field_scores      = Column(Text,    nullable=False)
    nlp_reasoning     = Column(Text,    nullable=False)
    issues            = Column(Text,    nullable=False)
    heatmap_path      = Column(String,  nullable=True)
    report_path       = Column(String,  nullable=True)
    processing_time_s = Column(Float,   nullable=False)
    created_at        = Column(DateTime, default=datetime.utcnow)
    user              = relationship("User", back_populates="verifications")


class Institution(Base):
    __tablename__ = "institutions"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    name          = Column(String,  unique=True, nullable=False, index=True)
    short_name    = Column(String,  nullable=True)
    code          = Column(String,  nullable=True)
    location      = Column(String,  nullable=True)
    accreditation = Column(String,  nullable=True)
    roll_pattern  = Column(String,  nullable=True)
    verified      = Column(Boolean, default=True)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============ SEED INSTITUTIONS ============
def seed_institutions(db: Session):
    if db.query(Institution).count() > 0:
        return
    try:
        from institution_list import INDIAN_UNIVERSITIES
        for uni in INDIAN_UNIVERSITIES:
            db.add(Institution(
                name          = uni["name"],
                short_name    = uni["short"],
                code          = uni["code"],
                location      = None,
                accreditation = uni["accreditation"],
                roll_pattern  = uni["roll_pattern"],
                verified      = True,
            ))
        db.commit()
        logger.info("Seeded %d institutions from institution_list.py", len(INDIAN_UNIVERSITIES))
    except ImportError:
        logger.warning("institution_list.py not found — seeding minimal fallback list")
        fallback = [
            {"name": "Indian Institute of Technology Bombay",   "short": "IIT Bombay",  "code": "IITB",   "accreditation": "NAAC A++", "roll_pattern": r"[0-9]{2}[A-Z]{1,2}[0-9]{4,6}"},
            {"name": "Indian Institute of Technology Delhi",    "short": "IIT Delhi",   "code": "IITD",   "accreditation": "NAAC A++", "roll_pattern": r"[A-Z]{3}[0-9]{6}[A-Z]"},
            {"name": "Indian Institute of Technology Madras",   "short": "IIT Madras",  "code": "IITM",   "accreditation": "NAAC A++", "roll_pattern": r"[A-Z]{2}[0-9]{2}[A-Z][0-9]{3}"},
            {"name": "National Institute of Technology Trichy", "short": "NIT Trichy",  "code": "NITT",   "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{6}[A-Z]{2}[0-9]{3}"},
            {"name": "University of Delhi",                     "short": "DU",          "code": "DU",     "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{7}"},
            {"name": "Anna University",                         "short": "AU",          "code": "AU",     "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{12}"},
            {"name": "BITS Pilani",                             "short": "BITS",        "code": "BITS",   "accreditation": "NAAC A",   "roll_pattern": r"20[0-9]{2}[A-Z]{2}[0-9]{4}[A-Z]"},
            {"name": "VIT University Vellore",                  "short": "VIT",         "code": "VIT",    "accreditation": "NAAC A++", "roll_pattern": r"[0-9]{2}[A-Z]{3}[0-9]{4}"},
            {"name": "Indian Institute of Science Bangalore",   "short": "IISc",        "code": "IISC",   "accreditation": "NAAC A++", "roll_pattern": r"[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{3}"},
            {"name": "Jawaharlal Nehru University",             "short": "JNU",         "code": "JNU",    "accreditation": "NAAC A++", "roll_pattern": r"[A-Z]{2}[0-9]{3}[A-Z][0-9]{3}"},
        ]
        for uni in fallback:
            db.add(Institution(
                name=uni["name"], short_name=uni["short"], code=uni["code"],
                accreditation=uni["accreditation"], roll_pattern=uni["roll_pattern"], verified=True,
            ))
        db.commit()


db_init = SessionLocal()
seed_institutions(db_init)
db_init.close()


# ============ PYDANTIC SCHEMAS ============
class RegisterRequest(BaseModel):
    email:     EmailStr
    password:  str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2)


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user:         Dict[str, Any]


class FieldScore(BaseModel):
    value:      str
    confidence: int = Field(..., ge=0, le=100)


class VerificationResponse(BaseModel):
    id:                str
    verdict:           str
    trust_score:       int
    forgery_score:     float
    field_confidence:  float
    nlp_anomaly_score: float
    institution_match: bool
    institution_name:  Optional[str]
    field_scores:      Dict[str, FieldScore]
    nlp_reasoning:     str
    issues:            List[str]
    heatmap_url:       Optional[str]
    report_url:        Optional[str]
    processing_time_s: float
    created_at:        str


class HistoryItem(BaseModel):
    id:               str
    filename:         str
    verdict:          str
    trust_score:      int
    institution_name: Optional[str]
    created_at:       str


class InstitutionItem(BaseModel):
    id:            int
    name:          str
    short_name:    Optional[str]
    code:          Optional[str]
    location:      Optional[str]
    accreditation: Optional[str]
    roll_pattern:  Optional[str]
    verified:      bool


class HealthResponse(BaseModel):
    status:    str
    version:   str
    timestamp: str
    model_loaded: bool


# ============ AUTH HELPERS ============
def hash_password(password: str) -> str:
    # bcrypt has a 72-byte limit — hash with SHA-256 first to handle long passwords safely
    import hashlib
    pw_bytes = hashlib.sha256(password.encode()).hexdigest().encode()
    return _bcrypt.hashpw(pw_bytes, _bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    import hashlib
    pw_bytes = hashlib.sha256(plain.encode()).hexdigest().encode()
    return _bcrypt.checkpw(pw_bytes, hashed.encode())


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    token = credentials.credentials
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# ============ REAL ML PIPELINE ============

def _pdf_to_image(pdf_path: Path) -> Path:
    """Convert first page of a PDF to a PNG and return its path."""
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=1)
        if not pages:
            raise ValueError("PDF has no pages")
        img_path = pdf_path.with_suffix(".png")
        pages[0].save(str(img_path), "PNG")
        return img_path
    except Exception as e:
        raise RuntimeError(f"PDF conversion failed: {e}")


def _build_reasoning(verdict: str, forgery_score: float, field_result: dict,
                     inst_match: dict, warnings: list[str]) -> str:
    """Generate a human-readable reasoning string from pipeline outputs."""
    lines = []

    # Forgery analysis
    if forgery_score < 0.3:
        lines.append(
            f"ELA analysis shows uniform compression across the document "
            f"(forgery score {forgery_score:.2f}) — no tampered regions detected."
        )
    elif forgery_score < 0.6:
        lines.append(
            f"Moderate ELA anomalies detected (forgery score {forgery_score:.2f}). "
            "Some regions show re-compression artifacts that may indicate localised edits."
        )
    else:
        lines.append(
            f"High ELA anomaly score ({forgery_score:.2f}) across multiple regions. "
            "Strong forensic indicators of document manipulation."
        )

    # Institution
    if inst_match.get("matched"):
        lines.append(
            f"Institution '{inst_match['name']}' verified in database "
            f"(accreditation: {inst_match.get('accreditation', 'N/A')})."
        )
    else:
        lines.append("Institution could not be matched against the verified database.")

    # Field warnings
    if warnings:
        lines.append("Field validation issues: " + "; ".join(warnings) + ".")

    # Overall
    if verdict == "GENUINE":
        lines.append("All checks passed. Certificate appears authentic.")
    elif verdict == "SUSPICIOUS":
        lines.append("Recommend manual review by a verification officer.")
    else:
        lines.append("Multiple forensic indicators suggest this certificate is not authentic.")

    return " ".join(lines)


def _build_issues(forgery_score: float, tamper_regions: list,
                  field_result: dict, inst_match: dict) -> list[str]:
    """Collect specific issues found during analysis."""
    issues = []

    if forgery_score > 0.6:
        issues.append(f"High forgery score ({forgery_score:.2f}) — ELA anomalies detected")
    if len(tamper_regions) > 0:
        issues.append(f"{len(tamper_regions)} tampered region(s) identified by ELA contour analysis")
    if not inst_match.get("matched"):
        issues.append("Institution not found in verified database")

    for w in field_result.get("warnings", []):
        issues.append(w)

    if not field_result.get("roll_valid") and inst_match.get("matched"):
        issues.append("Roll number format does not match institution's expected pattern")

    return issues


def _llm_reasoning(
    verdict: str,
    forgery_score: float,
    field_result: dict,
    inst_match: dict,
    issues: list[str],
) -> str:
    """
    Generate forensic reasoning using a local Ollama LLM (phi3 preferred).
    Falls back to rule-based reasoning if Ollama is unavailable.
    """
    try:
        import ollama

        preferred = ["phi3", "llama3", "mistral"]
        available = [m.model.split(":")[0] for m in ollama.list().models]
        model = next((m for m in preferred if m in available), None)

        if model is None:
            raise RuntimeError("No supported Ollama model found")

        fields = field_result.get("fields", {})
        field_lines = "\n".join(
            f"  - {k}: {v['value']} (confidence {int(v['confidence'] * 100)}%)"
            for k, v in fields.items()
        ) or "  - No fields extracted"

        institution  = inst_match.get("name", "Unknown") if inst_match.get("matched") else "Not found in verified database"
        issues_text  = "; ".join(issues) or "None"
        warning_text = "; ".join(field_result.get("warnings", [])) or "None"

        prompt = f"""You are a forensic document analyst specializing in academic certificate verification.

Analyze the following certificate verification result and write a concise 2-3 sentence professional forensic reasoning. Be specific. Do not repeat the verdict word.

Verdict: {verdict}
Forgery Score: {forgery_score:.2f} (0=genuine, 1=fake)
Field Confidence: {field_result.get('overall_confidence', 0):.2f}
Institution: {institution}
Extracted Fields:
{field_lines}
Detected Issues: {issues_text}
Field Warnings: {warning_text}

Write only the reasoning paragraph, 2-3 sentences maximum."""

        logger.info("Generating LLM reasoning with model: %s", model)
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 220},
        )
        reasoning = response["message"]["content"].strip()
        logger.info("LLM reasoning generated (%d chars)", len(reasoning))
        return reasoning

    except Exception as e:
        logger.warning("Ollama unavailable (%s) — using rule-based reasoning", e)
        return _build_reasoning(verdict, forgery_score, field_result, inst_match,
                                field_result.get("warnings", []))


def run_ml_pipeline(file_path: Path) -> Dict[str, Any]:
    """
    Full 4-step AI pipeline:
      1. PDF → image conversion (if needed)
      2. Forgery detection  (EfficientNet-B4 + ELA)
      3. Field extraction   (OCR + regex + institution matching)
      4. Trust score fusion + verdict
    """
    img_path = file_path
    converted_pdf = None

    # Step 1 — PDF conversion
    if file_path.suffix.lower() == ".pdf":
        img_path = _pdf_to_image(file_path)
        converted_pdf = img_path

    try:
        # Step 2 — Forgery detection
        logger.info("Running forgery detection on %s", img_path.name)
        forgery_result = predict(img_path, _forgery_model)
        forgery_score  = forgery_result["forgery_score"]          # 0=genuine, 1=fake
        gradcam        = forgery_result["gradcam_heatmap"]        # np.ndarray H×W×3
        tamper_regions = forgery_result["tamper_regions"]

        # Step 3 — Field extraction (OCR)
        logger.info("Running field extraction on %s", img_path.name)
        field_result      = _field_extractor.extract(img_path)
        extracted_fields  = field_result["fields"]
        inst_match        = field_result["institution_match"]
        field_confidence  = field_result["overall_confidence"]

        # Step 4 — Trust score fusion
        # forgery_score is P(fake): high = bad → invert for "genuine confidence"
        genuine_conf = 1.0 - forgery_score
        # field_confidence already in [0,1]
        # nlp_anomaly_score: use warnings count as a proxy (0 warnings = 1.0)
        n_warnings       = len(field_result.get("warnings", []))
        nlp_score        = max(0.0, 1.0 - n_warnings * 0.15)

        trust_raw   = genuine_conf * 0.45 + field_confidence * 0.35 + nlp_score * 0.20
        trust_score = int(trust_raw * 100)

        if trust_score >= 75:
            verdict = "GENUINE"
        elif trust_score >= 45:
            verdict = "SUSPICIOUS"
        else:
            verdict = "FAKE"

        # Build field_scores dict for API response (field_name → {value, confidence})
        field_scores_out: Dict[str, Dict] = {}
        field_map = {
            "student_name": "STUDENT NAME",
            "institution":  "INSTITUTION",
            "degree":       "DEGREE",
            "discipline":   "DISCIPLINE",
            "issue_date":   "ISSUE DATE",
            "grade":        "GRADE",
            "roll_number":  "ROLL NUMBER",
            "serial_number":"SERIAL NUMBER",
        }
        for key, label in field_map.items():
            if key in extracted_fields:
                f = extracted_fields[key]
                field_scores_out[label] = {
                    "value":      f["value"],
                    "confidence": int(f["confidence"] * 100),
                }

        # Fallback: if OCR found nothing, use placeholder
        if not field_scores_out:
            field_scores_out = {
                "STUDENT NAME": {"value": "Not extracted", "confidence": 0},
                "INSTITUTION":  {"value": "Not extracted", "confidence": 0},
            }

        issues    = _build_issues(forgery_score, tamper_regions, field_result, inst_match)
        reasoning = _llm_reasoning(verdict, forgery_score, field_result, inst_match, issues)

        return {
            "verdict":           verdict,
            "trust_score":       trust_score,
            "forgery_score":     round(forgery_score, 4),
            "field_confidence":  round(field_confidence, 4),
            "nlp_anomaly_score": round(nlp_score, 4),
            "institution_match": bool(inst_match.get("matched", False)),
            "institution_name":  inst_match.get("name") if inst_match.get("matched") else None,
            "field_scores":      field_scores_out,
            "nlp_reasoning":     reasoning,
            "issues":            issues,
            "gradcam":           gradcam,   # np.ndarray — saved separately
        }

    finally:
        # Clean up converted PDF image
        if converted_pdf and converted_pdf.exists():
            converted_pdf.unlink(missing_ok=True)


# ============ FASTAPI APP ============
app = FastAPI(
    title="CertiProof API",
    description="AI-powered certificate verification system",
    version=VERSION,
)

@app.on_event("startup")
async def startup_event():
    _load_models()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ ENDPOINTS ============

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status":       "ok",
        "version":      VERSION,
        "timestamp":    datetime.utcnow().isoformat(),
        "model_loaded": CHECKPOINT.exists(),
    }


@app.post("/api/v1/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user = User(
        email=req.email,
        full_name=req.full_name,
        hashed_password=hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.id})
    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id":         user.id,
            "email":      user.email,
            "full_name":  user.full_name,
            "created_at": user.created_at.isoformat(),
        },
    }


@app.post("/api/v1/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    token = create_access_token({"sub": user.id})
    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id":         user.id,
            "email":      user.email,
            "full_name":  user.full_name,
            "created_at": user.created_at.isoformat(),
        },
    }


@app.post("/api/v1/verify", response_model=VerificationResponse, status_code=status.HTTP_201_CREATED)
async def verify_certificate(
    file:         UploadFile = File(...),
    current_user: User       = Depends(get_current_user),
    db:           Session    = Depends(get_db),
):
    """Upload a certificate (PDF/JPG/PNG) and run the full AI verification pipeline."""
    allowed_types = {"application/pdf", "image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF, JPG, and PNG files are allowed",
        )

    # Save upload
    file_id  = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix or ".jpg"
    file_path = UPLOAD_DIR / f"{file_id}{file_ext}"

    content = await file.read()

    # Enforce 10 MB file size limit
    MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum allowed size is 10 MB (got {len(content) / 1024 / 1024:.1f} MB)",
        )

    file_path.write_bytes(content)

    # Run real ML pipeline
    start_time = time.time()
    try:
        result = run_ml_pipeline(file_path)
    except Exception as e:
        logger.error("ML pipeline failed for %s: %s", file.filename, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )
    processing_time = time.time() - start_time

    # Save GradCAM heatmap as JPEG
    heatmap_path: Optional[str] = None
    gradcam = result.pop("gradcam", None)
    if gradcam is not None:
        heatmap_file = HEATMAP_DIR / f"{file_id}_heatmap.jpg"
        heatmap_bgr  = cv2.cvtColor(gradcam, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(heatmap_file), heatmap_bgr)
        heatmap_path = str(heatmap_file)

    # Persist to DB
    verification = Verification(
        id=file_id,
        user_id=current_user.id,
        filename=file.filename,
        verdict=result["verdict"],
        trust_score=result["trust_score"],
        forgery_score=result["forgery_score"],
        field_confidence=result["field_confidence"],
        nlp_anomaly_score=result["nlp_anomaly_score"],
        institution_match=result["institution_match"],
        institution_name=result["institution_name"],
        field_scores=json.dumps(result["field_scores"]),
        nlp_reasoning=result["nlp_reasoning"],
        issues=json.dumps(result["issues"]),
        heatmap_path=heatmap_path,
        processing_time_s=processing_time,
    )
    db.add(verification)
    db.commit()
    db.refresh(verification)

    return _verification_to_response(verification)


@app.get("/api/v1/verify/{id}", response_model=VerificationResponse)
def get_verification(
    id:           str,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    v = db.query(Verification).filter(
        Verification.id == id, Verification.user_id == current_user.id
    ).first()
    if not v:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Verification not found")
    return _verification_to_response(v)


@app.get("/api/v1/history", response_model=List[HistoryItem])
def get_history(
    limit:        int     = Query(50, ge=1, le=100),
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    verifications = (
        db.query(Verification)
        .filter(Verification.user_id == current_user.id)
        .order_by(Verification.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":               v.id,
            "filename":         v.filename,
            "verdict":          v.verdict,
            "trust_score":      v.trust_score,
            "institution_name": v.institution_name,
            "created_at":       v.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        for v in verifications
    ]


@app.get("/api/v1/heatmap/{id}")
def get_heatmap(
    id:           str,
    token:        Optional[str] = Query(None),   # allow token via ?token= for <img> tags
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Return the GradCAM heatmap image for a verification."""
    v = db.query(Verification).filter(
        Verification.id == id, Verification.user_id == current_user.id
    ).first()
    if not v:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Verification not found")
    if not v.heatmap_path or not Path(v.heatmap_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Heatmap not available")
    return FileResponse(v.heatmap_path, media_type="image/jpeg")


@app.post("/api/v1/send-report/{id}", status_code=200)
def send_report_email(
    id:           str,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Send verification report email directly to the logged-in user via Gmail SMTP."""
    if not EMAIL_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Email not configured. Add SMTP_USER and SMTP_PASSWORD to .env"
        )

    v = db.query(Verification).filter(
        Verification.id == id, Verification.user_id == current_user.id
    ).first()
    if not v:
        raise HTTPException(status_code=404, detail="Verification not found")

    try:
        issues       = json.loads(v.issues) if v.issues else []
        field_scores = json.loads(v.field_scores) if v.field_scores else {}

        verdict_color = {"GENUINE": "#10b981", "SUSPICIOUS": "#f59e0b", "FAKE": "#ef4444"}.get(v.verdict, "#3b7bf8")
        vc_rgb = ",".join(str(int(verdict_color.lstrip("#")[i:i+2], 16)) for i in (0, 2, 4))

        fields_rows = "".join(
            f"""<tr>
                  <td style="padding:8px 12px;color:#4a5a8a;font-size:12px;border-bottom:1px solid #1a2340;">{k}</td>
                  <td style="padding:8px 12px;color:#dce8ff;font-size:12px;border-bottom:1px solid #1a2340;">{fv['value']}</td>
                  <td style="padding:8px 12px;font-size:12px;border-bottom:1px solid #1a2340;
                      color:{'#10b981' if fv['confidence']>80 else '#f59e0b' if fv['confidence']>60 else '#ef4444'};">
                    {fv['confidence']}%
                  </td>
                </tr>"""
            for k, fv in field_scores.items()
        ) or "<tr><td colspan='3' style='padding:8px 12px;color:#4a5a8a;'>No fields extracted</td></tr>"

        issues_html = "".join(
            f'<li style="color:#fca5a5;font-size:12px;margin-bottom:4px;">→ {iss}</li>'
            for iss in issues
        ) if issues else "<li style='color:#4a5a8a;font-size:12px;'>No anomalies detected</li>"

        anomalies_block = f"""
        <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);border-radius:10px;padding:16px 20px;margin-bottom:16px;">
          <p style="color:#ef4444;font-size:10px;letter-spacing:3px;margin:0 0 10px;">⚠ DETECTED ANOMALIES ({len(issues)})</p>
          <ul style="margin:0;padding-left:16px;">{issues_html}</ul>
        </div>""" if issues else ""

        inst_color = "#10b981" if v.institution_match else "#ef4444"
        inst_text  = ("✓ " + (v.institution_name or "Verified")) if v.institution_match else "✕ Not in verified database"

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#05080f;font-family:'Segoe UI',Arial,sans-serif;">
<div style="max-width:600px;margin:0 auto;padding:32px 16px;">

  <div style="background:linear-gradient(135deg,#0e1428,#0d1f3c);border:1px solid rgba(59,123,248,0.2);border-radius:12px;padding:28px;text-align:center;margin-bottom:20px;">
    <div style="font-size:32px;margin-bottom:8px;">🛡</div>
    <h1 style="color:#dce8ff;font-size:22px;margin:0 0 4px;letter-spacing:3px;">CERTIPROOF</h1>
    <p style="color:#4a5a8a;font-size:11px;margin:0;letter-spacing:2px;">FORENSIC CERTIFICATE ANALYSIS REPORT</p>
  </div>

  <div style="background:rgba(14,20,40,0.9);border:1px solid rgba(59,123,248,0.15);border-radius:10px;padding:20px;margin-bottom:16px;">
    <p style="color:#dce8ff;font-size:14px;margin:0 0 8px;">Hi <strong>{current_user.full_name}</strong>,</p>
    <p style="color:#4a5a8a;font-size:13px;margin:0;line-height:1.6;">
      Your certificate <strong style="color:#dce8ff;">{v.filename}</strong> has been analyzed.
      Here is your forensic verification report.
    </p>
  </div>

  <div style="background:rgba(14,20,40,0.9);border:1px solid rgba(59,123,248,0.15);border-radius:10px;padding:24px;margin-bottom:16px;text-align:center;">
    <p style="color:#4a5a8a;font-size:10px;letter-spacing:3px;margin:0 0 12px;">VERDICT</p>
    <div style="display:inline-block;padding:10px 32px;border-radius:999px;background:rgba({vc_rgb},0.15);border:1px solid {verdict_color};color:{verdict_color};font-size:18px;font-weight:700;letter-spacing:3px;margin-bottom:20px;">
      {v.verdict}
    </div>
    <table style="width:100%;border-collapse:collapse;">
      <tr>
        <td style="text-align:center;padding:8px;">
          <div style="color:{verdict_color};font-size:36px;font-weight:800;">{v.trust_score}</div>
          <div style="color:#4a5a8a;font-size:10px;letter-spacing:2px;">TRUST SCORE</div>
        </td>
        <td style="text-align:center;padding:8px;">
          <div style="color:#3b7bf8;font-size:36px;font-weight:800;">{int(v.forgery_score*100)}%</div>
          <div style="color:#4a5a8a;font-size:10px;letter-spacing:2px;">FORGERY SCORE</div>
        </td>
        <td style="text-align:center;padding:8px;">
          <div style="color:#a855f7;font-size:36px;font-weight:800;">{int(v.field_confidence*100)}%</div>
          <div style="color:#4a5a8a;font-size:10px;letter-spacing:2px;">FIELD CONFIDENCE</div>
        </td>
      </tr>
    </table>
  </div>

  <div style="background:rgba(14,20,40,0.9);border:1px solid rgba(59,123,248,0.15);border-radius:10px;padding:16px 20px;margin-bottom:16px;">
    <table style="width:100%;"><tr>
      <td style="color:#4a5a8a;font-size:11px;letter-spacing:2px;">INSTITUTION</td>
      <td style="text-align:right;color:{inst_color};font-size:13px;font-weight:600;">{inst_text}</td>
    </tr></table>
  </div>

  <div style="background:rgba(14,20,40,0.9);border:1px solid rgba(59,123,248,0.15);border-radius:10px;padding:20px;margin-bottom:16px;">
    <p style="color:#4a5a8a;font-size:10px;letter-spacing:3px;margin:0 0 14px;">EXTRACTED FIELDS</p>
    <table style="width:100%;border-collapse:collapse;">
      <thead><tr style="background:rgba(255,255,255,0.03);">
        <th style="padding:8px 12px;color:#4a5a8a;font-size:10px;text-align:left;border-bottom:1px solid #1a2340;">FIELD</th>
        <th style="padding:8px 12px;color:#4a5a8a;font-size:10px;text-align:left;border-bottom:1px solid #1a2340;">VALUE</th>
        <th style="padding:8px 12px;color:#4a5a8a;font-size:10px;text-align:left;border-bottom:1px solid #1a2340;">CONF.</th>
      </tr></thead>
      <tbody>{fields_rows}</tbody>
    </table>
  </div>

  {anomalies_block}

  <div style="background:rgba(14,20,40,0.9);border:1px solid rgba(59,123,248,0.15);border-radius:10px;padding:20px;margin-bottom:16px;">
    <p style="color:#4a5a8a;font-size:10px;letter-spacing:3px;margin:0 0 10px;">🧠 AI REASONING</p>
    <p style="color:#dce8ff;font-size:13px;line-height:1.7;margin:0;">{v.nlp_reasoning}</p>
  </div>

  <div style="text-align:center;padding:16px;">
    <p style="color:#4a5a8a;font-size:10px;letter-spacing:1px;margin:0;">
      🔒 CERTIPROOF · CONFIDENTIAL · AI-POWERED CERTIFICATE VERIFICATION<br>
      <span style="font-size:9px;">Analyzed on {v.created_at.strftime('%Y-%m-%d %H:%M UTC')} · {v.processing_time_s:.2f}s</span>
    </p>
  </div>

</div></body></html>"""

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"CertiProof Report — {v.filename} [{v.verdict}]"
        msg["From"]    = f"CertiProof <{SMTP_USER}>"
        msg["To"]      = current_user.email
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, current_user.email, msg.as_string())

        logger.info("Report email sent to %s", current_user.email)
        return {"message": f"Report sent to {current_user.email}"}

    except smtplib.SMTPAuthenticationError:
        raise HTTPException(status_code=401, detail="Email authentication failed. Check SMTP_USER and SMTP_PASSWORD in .env")
    except Exception as e:
        logger.error("Email send failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")



def get_report(
    id:           str,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Generate and return a PDF verification report."""
    v = db.query(Verification).filter(
        Verification.id == id, Verification.user_id == current_user.id
    ).first()
    if not v:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Verification not found")

    pdf_path = REPORT_DIR / f"{v.id}_report.pdf"

    # Generate if not cached
    if not pdf_path.exists():
        try:
            _generate_pdf_report(v, pdf_path)
        except Exception as e:
            logger.error("PDF generation failed: %s", e)
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"certiproof_{v.filename.rsplit('.', 1)[0]}_report.pdf",
    )


def _generate_pdf_report(v: "Verification", out_path: Path) -> None:
    """Generate a simple PDF report using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import pt
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    doc    = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=40*pt, rightMargin=40*pt, topMargin=40*pt, bottomMargin=40*pt)
    styles = getSampleStyleSheet()
    story  = []

    # Colours
    DARK   = colors.HexColor("#080c18")
    CARD   = colors.HexColor("#0e1428")
    BLUE   = colors.HexColor("#3b7bf8")
    GREEN  = colors.HexColor("#10b981")
    AMBER  = colors.HexColor("#f59e0b")
    RED    = colors.HexColor("#ef4444")
    TEXT   = colors.HexColor("#c9d4f0")
    MUTED  = colors.HexColor("#5a6a9a")

    verdict_color = GREEN if v.verdict == "GENUINE" else AMBER if v.verdict == "SUSPICIOUS" else RED

    title_style = ParagraphStyle("title", parent=styles["Title"], textColor=TEXT, backColor=DARK, fontSize=20, spaceAfter=6)
    h2_style    = ParagraphStyle("h2",    parent=styles["Heading2"], textColor=BLUE, fontSize=11, spaceBefore=14, spaceAfter=4)
    body_style  = ParagraphStyle("body",  parent=styles["Normal"],   textColor=TEXT, fontSize=9,  leading=14)
    muted_style = ParagraphStyle("muted", parent=styles["Normal"],   textColor=MUTED, fontSize=8)

    story.append(Paragraph("CERTIPROOF — FORENSIC ANALYSIS REPORT", title_style))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", muted_style))
    story.append(HRFlowable(width="100%", color=BLUE, thickness=1, spaceAfter=10))

    # Summary table
    verdict_str = f'<font color="#{("10b981" if v.verdict=="GENUINE" else "f59e0b" if v.verdict=="SUSPICIOUS" else "ef4444")}">{v.verdict}</font>'
    summary_data = [
        ["File",          v.filename],
        ["Verdict",       Paragraph(verdict_str, body_style)],
        ["Trust Score",   f"{v.trust_score} / 100"],
        ["Forgery Score", f"{v.forgery_score:.4f}"],
        ["Field Conf.",   f"{v.field_confidence:.4f}"],
        ["Institution",   v.institution_name or "Not matched"],
        ["Analyzed",      v.created_at.strftime("%Y-%m-%d %H:%M")],
        ["Processing",    f"{v.processing_time_s:.2f}s"],
    ]
    tbl = Table(summary_data, colWidths=[120*pt, 340*pt])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), CARD),
        ("TEXTCOLOR",   (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR",   (1, 0), (1, -1), TEXT),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [DARK, CARD]),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#1a2340")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",(0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0), (-1, -1), 5),
    ]))
    story.append(tbl)

    # Issues
    issues = json.loads(v.issues)
    if issues:
        story.append(Paragraph("DETECTED ANOMALIES", h2_style))
        for iss in issues:
            story.append(Paragraph(f"→ {iss}", ParagraphStyle("issue", parent=body_style, textColor=RED)))

    # Fields
    field_scores = json.loads(v.field_scores)
    if field_scores:
        story.append(Paragraph("EXTRACTED FIELDS", h2_style))
        rows = [["Field", "Value", "Confidence"]]
        for fname, fdata in field_scores.items():
            rows.append([fname, fdata["value"], f"{fdata['confidence']}%"])
        ftbl = Table(rows, colWidths=[130*pt, 240*pt, 90*pt])
        ftbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("TEXTCOLOR",   (0, 1), (-1, -1), TEXT),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [DARK, CARD]),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#1a2340")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0,0), (-1, -1), 4),
        ]))
        story.append(ftbl)

    # Reasoning
    story.append(Paragraph("AI REASONING", h2_style))
    story.append(Paragraph(v.nlp_reasoning, body_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", color=MUTED, thickness=0.5))
    story.append(Paragraph("CERTIPROOF · CONFIDENTIAL · AI-POWERED CERTIFICATE VERIFICATION", muted_style))

    doc.build(story)


@app.get("/api/v1/institutions", response_model=List[InstitutionItem])
def search_institutions(q: str = Query("", min_length=0), db: Session = Depends(get_db)):
    if not q:
        institutions = db.query(Institution).limit(20).all()
    else:
        institutions = db.query(Institution).filter(
            (Institution.name.ilike(f"%{q}%")) | (Institution.short_name.ilike(f"%{q}%"))
        ).limit(20).all()
    return [
        {
            "id":            inst.id,
            "name":          inst.name,
            "short_name":    inst.short_name,
            "code":          inst.code,
            "location":      inst.location,
            "accreditation": inst.accreditation,
            "roll_pattern":  inst.roll_pattern,
            "verified":      inst.verified,
        }
        for inst in institutions
    ]


# ============ HELPERS ============
def _verification_to_response(v: Verification) -> dict:
    return {
        "id":                v.id,
        "verdict":           v.verdict,
        "trust_score":       v.trust_score,
        "forgery_score":     v.forgery_score,
        "field_confidence":  v.field_confidence,
        "nlp_anomaly_score": v.nlp_anomaly_score,
        "institution_match": v.institution_match,
        "institution_name":  v.institution_name,
        "field_scores":      json.loads(v.field_scores),
        "nlp_reasoning":     v.nlp_reasoning,
        "issues":            json.loads(v.issues),
        "heatmap_url":       f"/api/v1/heatmap/{v.id}" if v.heatmap_path else None,
        "report_url":        f"/api/v1/report/{v.id}",
        "processing_time_s": round(v.processing_time_s, 2),
        "created_at":        v.created_at.strftime("%Y-%m-%d %H:%M"),
    }


# ============ CHAT ENDPOINT ============

class ChatRequest(BaseModel):
    message:         str = Field(..., min_length=1, max_length=1000)
    verification_id: Optional[str] = None


def _build_chat_context(verification_id: Optional[str], user_id: str, db: Session) -> str:
    """Build system context for the chatbot from a verification result if provided."""
    base_context = """You are CertiProof AI Assistant, an expert in academic certificate forensics and verification.
You help users understand certificate verification results, explain forensic concepts, and answer questions about institutions.
Be concise, professional, and helpful. Keep responses under 150 words unless a detailed explanation is needed."""

    if not verification_id:
        return base_context

    v = db.query(Verification).filter(
        Verification.id == verification_id,
        Verification.user_id == user_id
    ).first()

    if not v:
        return base_context

    field_scores = json.loads(v.field_scores) if v.field_scores else {}
    issues       = json.loads(v.issues) if v.issues else []

    fields_text = "\n".join(
        f"  - {k}: {val['value']} (confidence {val['confidence']}%)"
        for k, val in field_scores.items()
    ) or "  - No fields extracted"

    return f"""{base_context}

The user is asking about a specific certificate verification result:
- File: {v.filename}
- Verdict: {v.verdict}
- Trust Score: {v.trust_score}/100
- Forgery Score: {v.forgery_score:.2f} (0=genuine, 1=fake)
- Field Confidence: {v.field_confidence:.2f}
- Institution: {v.institution_name or 'Not matched'}
- Institution Verified: {'Yes' if v.institution_match else 'No'}
- Detected Issues: {'; '.join(issues) if issues else 'None'}
- Extracted Fields:
{fields_text}
- AI Reasoning: {v.nlp_reasoning}
- Analyzed: {v.created_at.strftime('%Y-%m-%d %H:%M')}

Answer questions about this result specifically."""


@app.post("/api/v1/chat")
async def chat(
    req:          ChatRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """
    Streaming chat endpoint powered by Ollama (phi3/llama3).
    Optionally accepts a verification_id to give the LLM context about a result.
    """
    system_context = _build_chat_context(req.verification_id, current_user.id, db)

    async def stream_response():
        try:
            import ollama

            preferred  = ["phi3", "llama3", "mistral"]
            available  = [m.model.split(":")[0] for m in ollama.list().models]
            model      = next((m for m in preferred if m in available), None)

            if model is None:
                yield "data: Ollama is running but no supported model found. Please run: ollama pull phi3\n\n"
                return

            logger.info("Chat request — model: %s, user: %s", model, current_user.email)

            stream = ollama.chat(
                model=model,
                messages=[
                    {"role": "system",  "content": system_context},
                    {"role": "user",    "content": req.message},
                ],
                stream=True,
                options={"temperature": 0.5, "num_predict": 300},
            )

            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    # SSE format: data: <token>\n\n
                    yield f"data: {json.dumps({'token': token})}\n\n"

            yield "data: [DONE]\n\n"

        except ImportError:
            yield f"data: {json.dumps({'token': 'Ollama Python client not installed. Run: pip install ollama'})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("Chat error: %s", e)
            error_msg = "Ollama is not running. Please start it with: ollama serve"
            yield f"data: {json.dumps({'token': error_msg})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ============ RUN ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
