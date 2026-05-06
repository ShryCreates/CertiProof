"""
CertValidator Backend — AI Certificate Verification System
FastAPI + SQLite + JWT Auth + Mock ML Pipeline
"""

import os
import uuid
import random
import time
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Query, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Text, Integer, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship, DeclarativeBase
from passlib.context import CryptContext
from jose import JWTError, jwt

# ============ CONFIG ============
VERSION = "2.4.1"
SECRET_KEY = "certvalidator-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

UPLOAD_DIR = Path("./uploads")
HEATMAP_DIR = Path("./heatmaps")
REPORT_DIR = Path("./reports")

for d in [UPLOAD_DIR, HEATMAP_DIR, REPORT_DIR]:
    d.mkdir(exist_ok=True)

# ============ DATABASE ============
DATABASE_URL = "sqlite:///./certvalidator.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    verifications = relationship("Verification", back_populates="user")


class Verification(Base):
    __tablename__ = "verifications"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    verdict = Column(String, nullable=False)  # GENUINE | SUSPICIOUS | FAKE
    trust_score = Column(Integer, nullable=False)
    forgery_score = Column(Float, nullable=False)
    field_confidence = Column(Float, nullable=False)
    nlp_anomaly_score = Column(Float, nullable=False)
    institution_match = Column(Boolean, nullable=False)
    institution_name = Column(String, nullable=True)
    field_scores = Column(Text, nullable=False)  # JSON string
    nlp_reasoning = Column(Text, nullable=False)
    issues = Column(Text, nullable=False)  # JSON string
    heatmap_path = Column(String, nullable=True)
    report_path = Column(String, nullable=True)
    processing_time_s = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="verifications")


class Institution(Base):
    __tablename__ = "institutions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False, index=True)
    short_name = Column(String, nullable=True)
    location = Column(String, nullable=True)
    verified = Column(Boolean, default=True)


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
    institutions = [
        {"name": "Indian Institute of Technology Bombay", "short_name": "IIT Bombay", "location": "Mumbai, Maharashtra"},
        {"name": "Indian Institute of Technology Delhi", "short_name": "IIT Delhi", "location": "New Delhi"},
        {"name": "Indian Institute of Technology Madras", "short_name": "IIT Madras", "location": "Chennai, Tamil Nadu"},
        {"name": "National Institute of Technology Trichy", "short_name": "NIT Trichy", "location": "Tiruchirappalli, Tamil Nadu"},
        {"name": "Delhi University", "short_name": "DU", "location": "New Delhi"},
        {"name": "Jawaharlal Nehru University", "short_name": "JNU", "location": "New Delhi"},
        {"name": "Visvesvaraya Technological University", "short_name": "VTU Belagavi", "location": "Belagavi, Karnataka"},
        {"name": "Anna University", "short_name": "Anna Univ", "location": "Chennai, Tamil Nadu"},
        {"name": "Birla Institute of Technology and Science", "short_name": "BITS Pilani", "location": "Pilani, Rajasthan"},
        {"name": "Indian Institute of Science", "short_name": "IISc Bangalore", "location": "Bangalore, Karnataka"},
    ]
    for inst in institutions:
        db.add(Institution(**inst))
    db.commit()


# Seed on startup
db_init = SessionLocal()
seed_institutions(db_init)
db_init.close()


# ============ PYDANTIC SCHEMAS ============
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class FieldScore(BaseModel):
    value: str
    confidence: int = Field(..., ge=0, le=100)


class VerificationResponse(BaseModel):
    id: str
    verdict: str
    trust_score: int
    forgery_score: float
    field_confidence: float
    nlp_anomaly_score: float
    institution_match: bool
    institution_name: Optional[str]
    field_scores: Dict[str, FieldScore]
    nlp_reasoning: str
    issues: List[str]
    heatmap_url: Optional[str]
    report_url: Optional[str]
    processing_time_s: float
    created_at: str


class HistoryItem(BaseModel):
    id: str
    filename: str
    verdict: str
    trust_score: int
    institution_name: Optional[str]
    created_at: str


class InstitutionItem(BaseModel):
    id: int
    name: str
    short_name: Optional[str]
    location: Optional[str]
    verified: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# ============ AUTH HELPERS ============
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> User:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# ============ MOCK ML PIPELINE ============
def mock_ml_pipeline(filename: str, file_path: Path) -> Dict[str, Any]:
    """
    Simulates the 6-step AI pipeline with random scores.
    In production, replace with actual ML models.
    """
    time.sleep(random.uniform(1.5, 3.0))  # Simulate processing time
    
    # Random scores
    forgery = random.uniform(0.15, 0.95)
    field = random.uniform(0.25, 0.92)
    nlp = random.uniform(0.18, 0.85)
    
    # Trust score calculation (weighted)
    trust_score = int((forgery * 0.45 + field * 0.35 + nlp * 0.20) * 100)
    
    # Institution match (80% chance if trust_score > 50)
    institution_match = trust_score > 50 and random.random() > 0.2
    
    # Verdict logic
    if trust_score >= 75:
        verdict = "GENUINE"
    elif trust_score >= 45:
        verdict = "SUSPICIOUS"
    else:
        verdict = "FAKE"
    
    # Mock extracted fields
    field_scores = {
        "STUDENT NAME": {"value": random.choice(["Arjun Mehta", "Priya Sharma", "Rohit Kumar", "Ananya Singh"]), "confidence": random.randint(70, 98)},
        "INSTITUTION": {"value": random.choice(["IIT Bombay", "Delhi University", "NIT Trichy", "VTU Belagavi"]), "confidence": random.randint(65, 98)},
        "DEGREE": {"value": random.choice(["B.Tech", "M.Tech", "B.A. (Hons)", "B.E."]), "confidence": random.randint(75, 96)},
        "DISCIPLINE": {"value": random.choice(["Computer Science", "Economics", "Mechanical Engineering", "AI & ML"]), "confidence": random.randint(68, 94)},
        "ISSUE DATE": {"value": f"{random.randint(1, 28)} {random.choice(['Jan', 'Feb', 'Mar', 'Jun', 'Jul', 'Aug'])} {random.randint(2020, 2024)}", "confidence": random.randint(60, 95)},
        "GRADE": {"value": f"{random.uniform(6.5, 9.5):.1f} CGPA", "confidence": random.randint(45, 96)},
        "ROLL NUMBER": {"value": f"{random.randint(100000, 999999)}", "confidence": random.randint(40, 97)},
    }
    
    # Issues based on verdict
    issues = []
    if verdict == "SUSPICIOUS":
        issues = random.sample([
            "Grade region shows ELA anomalies",
            "Roll number format mismatch",
            "Logo position deviation",
            "Font inconsistency detected",
            "Signature clarity below threshold"
        ], k=random.randint(1, 3))
    elif verdict == "FAKE":
        issues = random.sample([
            "Institution not in verified DB",
            "Implausible grade value",
            "Multiple ELA tamper regions",
            "Signature DPI mismatch",
            "Date format invalid",
            "Seal authenticity failed"
        ], k=random.randint(2, 4))
    
    # NLP reasoning
    reasoning_templates = {
        "GENUINE": "All extracted fields are internally consistent. The institution seal, signature DPI, and font kerning match the verified template. ELA analysis shows uniform compression with no tampered regions. Grade and roll number formats align with the institution's known schema.",
        "SUSPICIOUS": "Moderate compression artifacts detected near critical fields, indicating possible localized edits. Some field formats deviate from the institution's standard schema. Recommend manual review by a verification officer.",
        "FAKE": "High ELA anomaly scores across multiple regions including signature, seal, and grade fields. Institution verification failed or grade values are statistically improbable. Multiple forensic indicators suggest document manipulation."
    }
    nlp_reasoning = reasoning_templates[verdict]
    
    return {
        "verdict": verdict,
        "trust_score": trust_score,
        "forgery_score": forgery,
        "field_confidence": field,
        "nlp_anomaly_score": nlp,
        "institution_match": institution_match,
        "institution_name": field_scores["INSTITUTION"]["value"] if institution_match else None,
        "field_scores": field_scores,
        "nlp_reasoning": nlp_reasoning,
        "issues": issues,
    }


# ============ FASTAPI APP ============
app = FastAPI(
    title="CertValidator API",
    description="AI-powered certificate verification system",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ ENDPOINTS ============

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user"""
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    
    user = User(
        email=req.email,
        full_name=req.full_name,
        hashed_password=hash_password(req.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = create_access_token({"sub": user.id})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "created_at": user.created_at.isoformat()
        }
    }


@app.post("/api/v1/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Login and get JWT token"""
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    
    token = create_access_token({"sub": user.id})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "created_at": user.created_at.isoformat()
        }
    }


@app.post("/api/v1/verify", response_model=VerificationResponse, status_code=status.HTTP_201_CREATED)
async def verify_certificate(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload and verify a certificate (PDF/JPG/PNG).
    Runs ML pipeline and returns verification result.
    """
    # Validate file type
    allowed_types = ["application/pdf", "image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF, JPG, and PNG files are allowed")
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    file_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Run ML pipeline
    start_time = time.time()
    result = mock_ml_pipeline(file.filename, file_path)
    processing_time = time.time() - start_time
    
    # Create verification record
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
        processing_time_s=processing_time,
    )
    db.add(verification)
    db.commit()
    db.refresh(verification)
    
    # Return response
    return {
        "id": verification.id,
        "verdict": verification.verdict,
        "trust_score": verification.trust_score,
        "forgery_score": verification.forgery_score,
        "field_confidence": verification.field_confidence,
        "nlp_anomaly_score": verification.nlp_anomaly_score,
        "institution_match": verification.institution_match,
        "institution_name": verification.institution_name,
        "field_scores": json.loads(verification.field_scores),  # Convert back to dict
        "nlp_reasoning": verification.nlp_reasoning,
        "issues": json.loads(verification.issues),  # Convert back to list
        "heatmap_url": f"/api/v1/heatmap/{verification.id}",
        "report_url": f"/api/v1/report/{verification.id}",
        "processing_time_s": round(verification.processing_time_s, 2),
        "created_at": verification.created_at.strftime("%Y-%m-%d %H:%M"),
    }


@app.get("/api/v1/verify/{id}", response_model=VerificationResponse)
def get_verification(id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get a single verification result by UUID"""
    verification = db.query(Verification).filter(Verification.id == id, Verification.user_id == current_user.id).first()
    if not verification:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Verification not found")
    
    return {
        "id": verification.id,
        "verdict": verification.verdict,
        "trust_score": verification.trust_score,
        "forgery_score": verification.forgery_score,
        "field_confidence": verification.field_confidence,
        "nlp_anomaly_score": verification.nlp_anomaly_score,
        "institution_match": verification.institution_match,
        "institution_name": verification.institution_name,
        "field_scores": json.loads(verification.field_scores),
        "nlp_reasoning": verification.nlp_reasoning,
        "issues": json.loads(verification.issues),
        "heatmap_url": f"/api/v1/heatmap/{verification.id}",
        "report_url": f"/api/v1/report/{verification.id}",
        "processing_time_s": round(verification.processing_time_s, 2),
        "created_at": verification.created_at.strftime("%Y-%m-%d %H:%M"),
    }


@app.get("/api/v1/history", response_model=List[HistoryItem])
def get_history(
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's verification history"""
    verifications = db.query(Verification).filter(Verification.user_id == current_user.id).order_by(Verification.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": v.id,
            "filename": v.filename,
            "verdict": v.verdict,
            "trust_score": v.trust_score,
            "institution_name": v.institution_name,
            "created_at": v.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        for v in verifications
    ]


@app.get("/api/v1/heatmap/{id}")
def get_heatmap(id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return heatmap image file (mock placeholder)"""
    verification = db.query(Verification).filter(Verification.id == id, Verification.user_id == current_user.id).first()
    if not verification:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Verification not found")
    
    # In production, return actual heatmap image
    # For now, return a placeholder response
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Heatmap generation not yet implemented")


@app.get("/api/v1/report/{id}")
def get_report(id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return PDF report file (mock placeholder)"""
    verification = db.query(Verification).filter(Verification.id == id, Verification.user_id == current_user.id).first()
    if not verification:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Verification not found")
    
    # In production, generate and return PDF report
    # For now, return a placeholder response
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="PDF report generation not yet implemented")


@app.get("/api/v1/institutions", response_model=List[InstitutionItem])
def search_institutions(q: str = Query("", min_length=0), db: Session = Depends(get_db)):
    """Search institution database"""
    if not q:
        institutions = db.query(Institution).limit(20).all()
    else:
        institutions = db.query(Institution).filter(
            (Institution.name.ilike(f"%{q}%")) | (Institution.short_name.ilike(f"%{q}%"))
        ).limit(20).all()
    
    return [
        {
            "id": inst.id,
            "name": inst.name,
            "short_name": inst.short_name,
            "location": inst.location,
            "verified": inst.verified,
        }
        for inst in institutions
    ]


# ============ RUN ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
