"""
CertiProof — OCR & Field Extraction Module
==============================================
Extracts structured fields from academic certificate images using:
  - Tesseract OCR  : raw text extraction
  - Regex pipeline : pattern-based field parsing
  - LayoutLMv3 NER : transformer-based named entity recognition (optional)
  - FuzzyWuzzy     : institution name matching against verified DB

Input:  certificate image path (JPG / PNG / PDF-rendered page)
Output: dict of extracted fields with confidence scores + institution match
"""

import re
import logging
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("field_extractor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

# ── Optional heavy imports (graceful fallback if not installed) ───────────────
try:
    import pytesseract

    # Auto-detect Tesseract on Windows
    import os, sys
    if sys.platform == "win32":
        _win_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.environ.get("USERNAME", "")),
        ]
        for _p in _win_paths:
            if os.path.exists(_p):
                pytesseract.pytesseract.tesseract_cmd = _p
                logger.info("Tesseract found at: %s", _p)
                break
        else:
            logger.warning(
                "Tesseract not found in common Windows paths. "
                "Download from https://github.com/UB-Mannheim/tesseract/wiki "
                "and install to C:\\Program Files\\Tesseract-OCR\\"
            )

    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False
    logger.warning("pytesseract not installed — run: pip install pytesseract")

try:
    from fuzzywuzzy import fuzz, process as fuzz_process
    FUZZY_OK = True
except ImportError:
    FUZZY_OK = False
    logger.warning("fuzzywuzzy not installed — fuzzy institution matching disabled")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    LAYOUTLM_OK = True
except ImportError:
    LAYOUTLM_OK = False
    logger.warning("transformers/torch not installed — LayoutLMv3 NER disabled")

from institution_list import INDIAN_UNIVERSITIES, ALL_NAMES, ALL_SHORTS


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEGREE_KEYWORDS = [
    "B.Tech", "B.E.", "B.Sc", "B.A", "B.Com", "B.B.A", "B.C.A",
    "M.Tech", "M.E.", "M.Sc", "M.A", "M.Com", "M.B.A", "M.C.A",
    "Ph.D", "PhD", "Doctor of Philosophy",
    "Bachelor of Technology", "Bachelor of Engineering",
    "Bachelor of Science", "Bachelor of Arts",
    "Master of Technology", "Master of Engineering",
    "Master of Science", "Master of Business Administration",
    "Diploma", "Post Graduate Diploma",
]

DISCIPLINE_KEYWORDS = [
    "Computer Science", "Computer Science and Engineering",
    "Information Technology", "Electronics", "Electrical Engineering",
    "Mechanical Engineering", "Civil Engineering", "Chemical Engineering",
    "Biotechnology", "Aerospace Engineering", "Automobile Engineering",
    "Artificial Intelligence", "Machine Learning", "Data Science",
    "Economics", "Commerce", "Mathematics", "Physics", "Chemistry",
    "MBA", "Finance", "Marketing", "Human Resources",
]

MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"

# Regex patterns for each field
PATTERNS = {
    "student_name": [
        r"(?:This is to certify that|Certified that|awarded to|presented to|This certifies that)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})",
        r"(?:Name|Student Name|Name of Student)\s*[:\-]\s*([A-Z][a-zA-Z\s]{3,40})",
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$",
    ],
    "institution": [
        r"([A-Z][A-Za-z\s]+(?:University|Institute|College|Academy|Institution|School|Technology|Science)[A-Za-z\s]*)",
        r"(?:Issued by|Awarded by|Conferred by)\s+([A-Z][A-Za-z\s,]+)",
    ],
    "degree": [
        r"((?:Bachelor|Master|Doctor|Ph\.?D|B\.Tech|M\.Tech|B\.E\.|M\.E\.|B\.Sc|M\.Sc|B\.A|M\.A|B\.Com|M\.Com|MBA|BCA|MCA|Diploma)[A-Za-z\s\.]*)",
        r"(?:Degree|Programme|Program|Course)\s*[:\-]\s*([A-Za-z\s\.]+)",
        r"(?:awarded the degree of|degree of)\s+([A-Za-z\s\.]+)",
    ],
    "discipline": [
        r"(?:in|of)\s+((?:" + "|".join(re.escape(d) for d in DISCIPLINE_KEYWORDS) + r"))",
        r"(?:Branch|Specialization|Stream|Department|Field)\s*[:\-]\s*([A-Za-z\s&]+)",
        r"(?:in the field of|specializing in)\s+([A-Za-z\s&]+)",
    ],
    "issue_date": [
        rf"(?:Date|Issued on|Date of Issue|Awarded on)\s*[:\-]?\s*(\d{{1,2}}[\s/\-]{MONTHS}[\s/\-]\d{{4}})",
        rf"(\d{{1,2}}[\s/\-]{MONTHS}[\s/\-]\d{{4}})",
        r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})",
        rf"({MONTHS}\s+\d{{1,2}},?\s+\d{{4}})",
    ],
    "grade": [
        r"(?:CGPA|GPA|Grade Point Average)\s*[:\-]?\s*(\d{1,2}\.\d{1,2})\s*(?:/\s*10)?",
        r"(?:Percentage|Marks)\s*[:\-]?\s*(\d{2,3}(?:\.\d{1,2})?)\s*%?",
        r"(?:Grade|Division)\s*[:\-]?\s*(First|Second|Third|Distinction|Pass|A\+?|B\+?|C\+?)",
        r"(\d{1,2}\.\d{2})\s*CGPA",
        r"(\d{2,3}(?:\.\d{1,2})?)\s*(?:out of|/)\s*100",
    ],
    "roll_number": [
        r"(?:Roll No|Roll Number|Enrollment No|Enrolment No|Registration No|Reg\.?\s*No)\s*[:\-\.]\s*([A-Z0-9]{5,15})",
        r"(?:Student ID|ID No|Exam Roll)\s*[:\-\.]\s*([A-Z0-9]{5,15})",
    ],
    "serial_number": [
        r"(?:Certificate No|Cert\.?\s*No|Serial No|Document No)\s*[:\-\.]\s*([A-Z0-9\-/]{4,20})",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING FOR OCR
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_for_ocr(image_path: str | Path) -> np.ndarray:
    """
    Enhance image quality for better Tesseract OCR accuracy.

    Steps: grayscale → denoise → adaptive threshold → deskew
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Upscale small images (Tesseract works best at 300 DPI equivalent)
    h, w = img.shape[:2]
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold — handles uneven lighting
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Deskew using moments
    coords = np.column_stack(np.where(binary < 128))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5:
            (h2, w2) = binary.shape
            M = cv2.getRotationMatrix2D((w2 // 2, h2 // 2), angle, 1.0)
            binary = cv2.warpAffine(binary, M, (w2, h2),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=255)

    return binary


# ═══════════════════════════════════════════════════════════════════════════════
# OCR
# ═══════════════════════════════════════════════════════════════════════════════

def run_ocr(image_path: str | Path) -> str:
    """
    Run Tesseract OCR on a certificate image.
    Returns raw extracted text, or empty string if Tesseract unavailable.
    """
    if not TESSERACT_OK:
        logger.warning("Tesseract not available — returning empty text")
        return ""

    try:
        processed = preprocess_for_ocr(image_path)
        pil_img = Image.fromarray(processed)
        config = "--oem 3 --psm 6 -l eng"
        text = pytesseract.image_to_string(pil_img, config=config)
        logger.debug("OCR extracted %d characters", len(text))
        return text
    except Exception as e:
        logger.error("OCR failed: %s", e)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# REGEX FIELD EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _confidence_from_match(match_text: str, field: str) -> float:
    """
    Heuristic confidence score based on match quality.
    - Longer, cleaner matches score higher
    - Known keyword matches get a boost
    """
    base = min(0.95, 0.5 + len(match_text.strip()) * 0.01)

    if field == "degree" and any(kw.lower() in match_text.lower() for kw in DEGREE_KEYWORDS):
        base = min(0.97, base + 0.15)
    if field == "discipline" and any(kw.lower() in match_text.lower() for kw in DISCIPLINE_KEYWORDS):
        base = min(0.97, base + 0.10)
    if field == "issue_date" and re.search(r"\d{4}", match_text):
        base = min(0.95, base + 0.10)
    if field == "grade" and re.search(r"\d", match_text):
        base = min(0.95, base + 0.10)

    return round(base, 4)


def extract_fields_regex(text: str) -> dict[str, dict]:
    """
    Extract certificate fields from OCR text using regex patterns.

    Returns
    -------
    dict mapping field_name → {"value": str, "confidence": float}
    """
    results: dict[str, dict] = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for field, pattern_list in PATTERNS.items():
        for pattern in pattern_list:
            # Try full text first
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                if len(value) >= 2:
                    results[field] = {
                        "value":      value,
                        "confidence": _confidence_from_match(value, field),
                        "method":     "regex",
                    }
                    break

        # Fallback: scan line by line for short patterns
        if field not in results:
            for line in lines:
                for pattern in pattern_list:
                    m = re.search(pattern, line, re.IGNORECASE)
                    if m:
                        value = m.group(1).strip()
                        if len(value) >= 2:
                            results[field] = {
                                "value":      value,
                                "confidence": _confidence_from_match(value, field) * 0.85,
                                "method":     "regex_line",
                            }
                            break
                if field in results:
                    break

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# INSTITUTION MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def match_institution(extracted_name: str) -> dict:
    """
    Match an extracted institution name against the verified university DB.

    Uses fuzzy string matching (fuzzywuzzy) if available,
    falls back to simple substring matching.

    Returns
    -------
    dict with keys: matched, name, short, code, accreditation,
                    match_score, roll_pattern
    """
    if not extracted_name:
        return {"matched": False, "name": None, "match_score": 0.0}

    # Try exact substring match first (fast path)
    name_lower = extracted_name.lower()
    for uni in INDIAN_UNIVERSITIES:
        if (uni["name"].lower() in name_lower or
                uni["short"].lower() in name_lower or
                uni["code"].lower() in name_lower):
            return {
                "matched":       True,
                "name":          uni["name"],
                "short":         uni["short"],
                "code":          uni["code"],
                "accreditation": uni["accreditation"],
                "match_score":   1.0,
                "roll_pattern":  uni["roll_pattern"],
            }

    # Fuzzy match
    if FUZZY_OK:
        all_candidates = ALL_NAMES + ALL_SHORTS
        best_match, score = fuzz_process.extractOne(
            extracted_name, all_candidates, scorer=fuzz.token_sort_ratio
        )
        if score >= 70:
            # Find the university entry for this match
            uni = next(
                (u for u in INDIAN_UNIVERSITIES
                 if u["name"] == best_match or u["short"] == best_match),
                None
            )
            if uni:
                return {
                    "matched":       True,
                    "name":          uni["name"],
                    "short":         uni["short"],
                    "code":          uni["code"],
                    "accreditation": uni["accreditation"],
                    "match_score":   round(score / 100, 4),
                    "roll_pattern":  uni["roll_pattern"],
                }

    return {"matched": False, "name": extracted_name, "match_score": 0.0}


def validate_roll_number(roll_number: str, institution_info: dict) -> bool:
    """Check if a roll number matches the institution's expected pattern."""
    if not institution_info.get("matched") or not roll_number:
        return False
    pattern = institution_info.get("roll_pattern", "")
    if not pattern:
        return False
    return bool(re.fullmatch(pattern, roll_number.strip()))


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUTLMV3 NER (optional — requires transformers + GPU recommended)
# ═══════════════════════════════════════════════════════════════════════════════

class LayoutLMv3Extractor:
    """
    Uses microsoft/layoutlmv3-base for document NER.
    Falls back gracefully if model unavailable or transformers not installed.

    Label mapping expected from fine-tuned model:
        B-NAME, I-NAME       → student_name
        B-ORG,  I-ORG        → institution
        B-DEGREE             → degree
        B-DATE,  I-DATE      → issue_date
        B-GRADE              → grade
        B-ROLLNO             → roll_number
    """

    # Map NER labels → our field names
    LABEL_MAP = {
        "B-NAME":   "student_name", "I-NAME":   "student_name",
        "B-ORG":    "institution",  "I-ORG":    "institution",
        "B-DEGREE": "degree",
        "B-DATE":   "issue_date",   "I-DATE":   "issue_date",
        "B-GRADE":  "grade",
        "B-ROLLNO": "roll_number",
    }

    def __init__(self, model_name: str = "microsoft/layoutlmv3-base") -> None:
        self.available = False
        if not LAYOUTLM_OK:
            logger.warning("LayoutLMv3 unavailable — transformers not installed")
            return
        try:
            logger.info("Loading LayoutLMv3 model: %s", model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model     = AutoModelForTokenClassification.from_pretrained(model_name)
            self.pipe      = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
            )
            self.available = True
            logger.info("LayoutLMv3 loaded successfully")
        except Exception as e:
            logger.warning("LayoutLMv3 load failed (%s) — NER disabled", e)

    def extract(self, text: str) -> dict[str, dict]:
        """Run NER on OCR text and return field dict."""
        if not self.available or not text.strip():
            return {}

        try:
            entities = self.pipe(text[:512])  # token limit
        except Exception as e:
            logger.error("LayoutLMv3 inference failed: %s", e)
            return {}

        results: dict[str, dict] = {}
        for ent in entities:
            field = self.LABEL_MAP.get(ent.get("entity_group", ""))
            if not field:
                continue
            score = round(float(ent.get("score", 0.0)), 4)
            word  = ent.get("word", "").strip()
            if not word:
                continue
            # Keep highest-confidence entity per field
            if field not in results or score > results[field]["confidence"]:
                results[field] = {
                    "value":      word,
                    "confidence": score,
                    "method":     "layoutlmv3",
                }

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# FIELD MERGER — combine regex + NER results
# ═══════════════════════════════════════════════════════════════════════════════

def merge_extractions(
    regex_fields: dict[str, dict],
    ner_fields:   dict[str, dict],
) -> dict[str, dict]:
    """
    Merge regex and NER results, keeping the higher-confidence value per field.
    NER results get a small boost (0.05) since they use contextual understanding.
    """
    merged = dict(regex_fields)

    for field, ner_val in ner_fields.items():
        boosted_conf = min(0.99, ner_val["confidence"] + 0.05)
        ner_boosted  = {**ner_val, "confidence": boosted_conf}

        if field not in merged:
            merged[field] = ner_boosted
        elif boosted_conf > merged[field]["confidence"]:
            merged[field] = ner_boosted

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS: FieldExtractor
# ═══════════════════════════════════════════════════════════════════════════════

class FieldExtractor:
    """
    Full OCR + field extraction pipeline for academic certificates.

    Usage
    -----
    extractor = FieldExtractor(use_layoutlmv3=False)  # True requires GPU
    result    = extractor.extract("certificate.jpg")
    """

    def __init__(self, use_layoutlmv3: bool = False, layoutlmv3_model: str = "microsoft/layoutlmv3-base") -> None:
        self.use_layoutlmv3 = use_layoutlmv3 and LAYOUTLM_OK
        self._ner: Optional[LayoutLMv3Extractor] = None

        if self.use_layoutlmv3:
            self._ner = LayoutLMv3Extractor(layoutlmv3_model)
            if not self._ner.available:
                self.use_layoutlmv3 = False

    # ── public API ────────────────────────────────────────────────────────────

    def extract(self, image_path: str | Path) -> dict:
        """
        Full extraction pipeline.

        Parameters
        ----------
        image_path : str | Path
            Path to certificate image (JPG / PNG).

        Returns
        -------
        dict with keys:
            fields            : dict of extracted fields (value + confidence)
            institution_match : institution DB lookup result
            roll_valid        : bool — roll number matches institution pattern
            raw_text          : raw OCR text
            overall_confidence: float — mean confidence across all fields
            warnings          : list of validation warnings
        """
        image_path = Path(image_path)
        logger.info("Extracting fields from: %s", image_path.name)

        # ── Step 1: OCR ───────────────────────────────────────────────────────
        raw_text = run_ocr(image_path)
        if not raw_text.strip():
            logger.warning("OCR returned empty text for %s", image_path.name)

        # ── Step 2: Regex extraction ──────────────────────────────────────────
        regex_fields = extract_fields_regex(raw_text)

        # ── Step 3: LayoutLMv3 NER (optional) ────────────────────────────────
        ner_fields: dict[str, dict] = {}
        if self.use_layoutlmv3 and self._ner:
            ner_fields = self._ner.extract(raw_text)

        # ── Step 4: Merge ─────────────────────────────────────────────────────
        fields = merge_extractions(regex_fields, ner_fields)

        # ── Step 5: Institution matching ──────────────────────────────────────
        inst_text = fields.get("institution", {}).get("value", "")
        institution_match = match_institution(inst_text)

        # Update institution field confidence based on DB match
        if institution_match["matched"] and "institution" in fields:
            boosted = min(0.99, fields["institution"]["confidence"] + institution_match["match_score"] * 0.1)
            fields["institution"]["confidence"] = round(boosted, 4)
            fields["institution"]["db_name"]    = institution_match["name"]

        # ── Step 6: Roll number validation ───────────────────────────────────
        roll_text  = fields.get("roll_number", {}).get("value", "")
        roll_valid = validate_roll_number(roll_text, institution_match)

        if "roll_number" in fields:
            if roll_valid:
                fields["roll_number"]["confidence"] = min(0.99, fields["roll_number"]["confidence"] + 0.10)
            else:
                fields["roll_number"]["confidence"] = max(0.10, fields["roll_number"]["confidence"] - 0.20)

        # ── Step 7: Grade sanity check ────────────────────────────────────────
        warnings = self._validate_fields(fields)

        # ── Step 8: Overall confidence ────────────────────────────────────────
        confs = [v["confidence"] for v in fields.values() if isinstance(v, dict) and "confidence" in v]
        overall_confidence = round(sum(confs) / len(confs), 4) if confs else 0.0

        logger.info(
            "Extracted %d fields | overall_conf=%.4f | institution_matched=%s",
            len(fields), overall_confidence, institution_match["matched"]
        )

        return {
            "fields":             fields,
            "institution_match":  institution_match,
            "roll_valid":         roll_valid,
            "raw_text":           raw_text,
            "overall_confidence": overall_confidence,
            "warnings":           warnings,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _validate_fields(self, fields: dict[str, dict]) -> list[str]:
        """Run sanity checks and return a list of warning strings."""
        warnings: list[str] = []

        # Grade range check
        grade_val = fields.get("grade", {}).get("value", "")
        cgpa_match = re.search(r"(\d{1,2}\.\d{1,2})", grade_val)
        if cgpa_match:
            cgpa = float(cgpa_match.group(1))
            if cgpa > 10.0:
                warnings.append(f"CGPA {cgpa} exceeds maximum (10.0) — likely tampered")
            elif cgpa < 1.0:
                warnings.append(f"CGPA {cgpa} is implausibly low")

        pct_match = re.search(r"(\d{2,3}(?:\.\d{1,2})?)\s*%", grade_val)
        if pct_match:
            pct = float(pct_match.group(1))
            if pct > 100.0:
                warnings.append(f"Percentage {pct}% exceeds 100 — likely tampered")

        # Date sanity check
        date_val = fields.get("issue_date", {}).get("value", "")
        year_match = re.search(r"(19|20)\d{2}", date_val)
        if year_match:
            year = int(year_match.group())
            if year < 1950 or year > 2030:
                warnings.append(f"Issue year {year} is outside plausible range (1950–2030)")

        # Name sanity check
        name_val = fields.get("student_name", {}).get("value", "")
        if name_val and len(name_val.split()) < 2:
            warnings.append("Student name appears incomplete (single word)")

        return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python field_extractor.py <image_path> [--layoutlmv3]")
        sys.exit(1)

    img_path       = sys.argv[1]
    use_lmv3       = "--layoutlmv3" in sys.argv

    extractor = FieldExtractor(use_layoutlmv3=use_lmv3)
    result    = extractor.extract(img_path)

    print("\n── Extracted Fields ───────────────────────────────────────")
    for field, info in result["fields"].items():
        print(f"  {field:<20} : {info['value']:<35} (conf={info['confidence']:.2f}, method={info.get('method','?')})")

    print(f"\n── Institution Match ──────────────────────────────────────")
    im = result["institution_match"]
    print(f"  Matched     : {im['matched']}")
    if im["matched"]:
        print(f"  Name        : {im['name']}")
        print(f"  Code        : {im['code']}")
        print(f"  Accreditation: {im['accreditation']}")
        print(f"  Match Score : {im['match_score']:.4f}")

    print(f"\n── Summary ────────────────────────────────────────────────")
    print(f"  Roll Valid          : {result['roll_valid']}")
    print(f"  Overall Confidence  : {result['overall_confidence']:.4f}")
    if result["warnings"]:
        print(f"  Warnings ({len(result['warnings'])}):")
        for w in result["warnings"]:
            print(f"    ⚠ {w}")

    # Save JSON output
    out_path = Path(img_path).stem + "_fields.json"
    with open(out_path, "w") as f:
        # Remove non-serialisable keys
        out = {**result}
        out.pop("raw_text", None)
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {out_path}")
