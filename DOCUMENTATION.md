
# CertiProof — Complete Project Documentation

> AI-Based Academic Certificate Authenticity Validator
> Version 2.5.0 | Python 3.14 · React 18 · FastAPI · PyTorch

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [ML Layer](#4-ml-layer)
5. [Backend API](#5-backend-api)
6. [Frontend](#6-frontend)
7. [Setup & Installation](#7-setup--installation)
8. [Running the Application](#8-running-the-application)
9. [API Reference](#9-api-reference)
10. [Training the Model](#10-training-the-model)
11. [How the AI Pipeline Works](#11-how-the-ai-pipeline-works)
12. [Database Schema](#12-database-schema)
13. [Configuration & Environment](#13-configuration--environment)
14. [Known Limitations](#14-known-limitations)

---

## 1. Project Overview

### Problem Statement

The increasing number of fake and manipulated academic certificates has become a major concern for educational institutions, employers, and government agencies. There is no unified and reliable system to verify the authenticity of these documents efficiently.

### Solution

CertiProof is a full-stack AI system that accepts a certificate image (JPG, PNG, or PDF) and runs it through a multi-stage forensic pipeline to determine whether it is GENUINE, SUSPICIOUS, or FAKE.

### Key Features

- EfficientNet-B4 deep learning model trained on real/fake certificate data
- Error Level Analysis (ELA) for detecting image tampering
- GradCAM heatmap showing which regions influenced the verdict
- Tesseract OCR for extracting structured fields (name, institution, grade, etc.)
- Fuzzy institution matching against a database of 50 Indian universities
- Roll number format validation per institution
- JWT-authenticated REST API
- React frontend with dark forensic UI
- Server-side PDF report generation

---

## 2. Architecture

```
FRONTEND (React + TypeScript)
  Login → Upload → Loading Animation → Result + Heatmap
  History Page · PDF Download · Share Result
         |
         | HTTP (JWT Bearer)
         v
BACKEND (FastAPI + SQLite)
  Auth (register/login) · /verify · /history
  /heatmap/{id} · /report/{id} · /institutions
         |
         | Python imports
         v
ML LAYER (PyTorch)
  forgery_detector.py     field_extractor.py
  - preprocess()          - preprocess_for_ocr()
  - generate_ela()        - run_ocr() [Tesseract]
  - build_model()         - extract_fields_regex()
  - generate_gradcam()    - match_institution()
  - predict()             - FieldExtractor class

  institution_list.py     train.py
  - 50 Indian universities - Training pipeline
```

### Trust Score Formula

```
trust_score = (genuine_confidence x 0.45)
            + (field_confidence   x 0.35)
            + (nlp_score          x 0.20)

Verdict:  >= 75  GENUINE
          >= 45  SUSPICIOUS
          <  45  FAKE
```

---

## 3. Project Structure

```
certguard-ai/
├── index.html                  # App entry point
├── package.json                # Frontend dependencies
├── src/
│   ├── main.tsx                # React root mount
│   ├── App.tsx                 # Router setup
│   ├── pages/
│   │   └── Index.tsx           # Main SPA (all pages)
│   └── lib/
│       └── api.ts              # API client
├── backend/
│   ├── main.py                 # FastAPI app + ML pipeline
│   ├── requirements.txt        # Python dependencies
│   ├── test_api.py             # Manual API tests
│   ├── .env.example            # Environment template
│   ├── uploads/                # Uploaded certificates
│   ├── heatmaps/               # GradCAM images
│   ├── reports/                # PDF reports
│   └── CertiProof.db        # SQLite database
└── ml/
    ├── forgery_detector.py     # Core ML model
    ├── train.py                # Training pipeline
    ├── field_extractor.py      # OCR + field extraction
    ├── institution_list.py     # 50 universities database
    ├── test_extractor.py       # 28 unit tests
    ├── sample_output.json      # Example response shape
    ├── requirements.txt        # ML dependencies
    ├── checkpoints/
    │   └── best_model.pth      # Trained weights
    ├── mlruns/                 # MLflow tracking
    └── training_data/
        ├── genuine/images/     # 1,008 genuine images
        └── fake/images/        # 1,153 fake images
```

---

## 4. ML Layer

### 4.1 forgery_detector.py

**Model Architecture**

| Component | Detail |
|---|---|
| Backbone | EfficientNet-B4 (pretrained ImageNet via timm) |
| Input | 6 channels (3 RGB + 3 ELA) at 224x224 |
| Head | Dropout → Linear(1792,512) → GELU → Dropout → Linear(512,2) |
| Output | Logits for [fake, genuine] |
| Parameters | 18,468,954 trainable |

**Preprocessing Steps**

1. Load with OpenCV
2. Detect skew via Hough line transform
3. Deskew if angle > 0.5 degrees
4. Crop 2% borders
5. CLAHE histogram equalisation per channel
6. Resize preserving aspect ratio
7. Pad to 224x224 with white background
8. Normalise to float32 [0,1]

**Error Level Analysis (ELA)**

Re-saves the image at JPEG quality 90, computes the absolute pixel difference amplified 10x. Tampered regions show higher differences because they were compressed at a different level than the original document.

**GradCAM**

Computes gradients of the predicted class score with respect to the last EfficientNet feature block. Produces a JET colormap heatmap overlaid on the original image.

**Usage:**
```python
from forgery_detector import build_model, predict
import torch

model = build_model()
ckpt = torch.load("checkpoints/best_model.pth", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

result = predict("certificate.jpg", model)
# result["forgery_score"]    float 0-1 (0=genuine, 1=fake)
# result["gradcam_heatmap"]  np.ndarray H x W x 3 uint8
# result["tamper_regions"]   list of {x, y, w, h, confidence}
```

---

### 4.2 train.py

**Training Configuration**

| Setting | Value |
|---|---|
| Split | 70% train / 15% val / 15% test (stratified) |
| Optimiser | AdamW lr=3e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR T_max=15 |
| Loss | CrossEntropyLoss + class weights + label smoothing 0.05 |
| Augmentations | HorizontalFlip, Rotation±5°, ColorJitter, RandomErasing |
| Checkpoint | Best by validation AUC |
| Tracking | MLflow in ml/mlruns/ |

**Run training:**
```bash
cd certguard-ai/ml
python train.py
```

**Monitor with MLflow:**
```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

---

### 4.3 field_extractor.py

**Extraction Pipeline**

1. Preprocess image (upscale, denoise, adaptive threshold, deskew)
2. Tesseract OCR (PSM 6, OEM 3)
3. Regex patterns for 8 fields
4. Fuzzy institution matching (fuzzywuzzy, threshold 70%)
5. Roll number regex validation per institution
6. Sanity checks (CGPA > 10, impossible dates, single-word names)

**Extracted Fields:** student_name, institution, degree, discipline, issue_date, grade, roll_number, serial_number

**Usage:**
```python
from field_extractor import FieldExtractor

extractor = FieldExtractor(use_layoutlmv3=False)
result = extractor.extract("certificate.jpg")
# result["fields"]             dict of field -> {value, confidence}
# result["institution_match"]  {matched, name, code, accreditation, ...}
# result["roll_valid"]         bool
# result["warnings"]           list of anomaly strings
# result["overall_confidence"] float 0-1
```

---

### 4.4 institution_list.py

50 Indian universities including IITs, NITs, DU, JNU, VIT, SRM, BITS, Amity, Manipal, and more. Each entry has: name, short name, code, NAAC accreditation grade, roll number regex pattern.

---

## 5. Backend API

### Stack

- FastAPI 0.115+ with Uvicorn
- SQLite via SQLAlchemy 2.0
- JWT auth (HS256, 7-day tokens) via python-jose
- bcrypt 5.x for password hashing
- reportlab for PDF generation

### Startup Sequence

1. Create SQLite tables
2. Seed 50 institutions from institution_list.py
3. Load EfficientNet-B4 checkpoint (once, on startup event)
4. Initialise FieldExtractor with Tesseract

---

## 6. Frontend

### Stack

- React 18 + TypeScript
- Vite 5 build tool
- Inline CSS with design tokens
- Fonts: Space Grotesk, JetBrains Mono
- jsPDF + QRCode for client-side PDF
- Native fetch with JWT Bearer headers

### Design Tokens

```
bg:       #080c18   (dark navy background)
card:     #0e1428   (card background)
blue:     #3b7bf8   (primary accent)
green:    #10b981   (GENUINE verdict)
amber:    #f59e0b   (SUSPICIOUS verdict)
red:      #ef4444   (FAKE verdict)
text:     #c9d4f0   (primary text)
muted:    #5a6a9a   (secondary text)
```

### API Client (src/lib/api.ts)

```typescript
login(email, password)           AuthResponse
register(email, password, name)  AuthResponse
verifyCertificate(file)          VerificationResult  // 3-min timeout
getVerification(id)              VerificationResult
getHistory(limit?)               HistoryItem[]
fetchHeatmapBlob(id)             string | null       // blob URL
```

Token stored in localStorage as `cv_token`.

---

## 7. Setup & Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| Tesseract OCR | 5.x |

### Install Tesseract (Windows)

Download from: https://github.com/UB-Mannheim/tesseract/wiki
Install to: C:\Program Files\Tesseract-OCR\
The app auto-detects this path.

### Install Python Dependencies

Backend:
```bash
cd certguard-ai/backend
pip install fastapi "uvicorn[standard]" sqlalchemy "pydantic[email]" \
            "python-jose[cryptography]" bcrypt python-multipart \
            aiofiles pdf2image opencv-python torch reportlab
```

ML:
```bash
cd certguard-ai/ml
pip install torch torchvision timm opencv-python Pillow numpy \
            scikit-learn mlflow tqdm pytesseract fuzzywuzzy \
            python-Levenshtein
```

### Install Frontend Dependencies

```bash
cd certguard-ai
npm install
```

---

## 8. Running the Application

### Start Backend

```bash
cd certguard-ai/backend
python main.py
```

Runs at http://localhost:8000
API docs at http://localhost:8000/docs

Note: First startup takes ~20 seconds while EfficientNet-B4 weights download from HuggingFace Hub.

### Start Frontend

```bash
cd certguard-ai
npm run dev
```

Opens at http://localhost:5173

### Environment Variables

Copy backend/.env.example to backend/.env:
```
SECRET_KEY=your-random-secret-here
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:8080
```

---

## 9. API Reference

All endpoints except /health and /api/v1/institutions require:
```
Authorization: Bearer <jwt_token>
```

### POST /api/v1/auth/register
Register a new user.

Request body:
```json
{ "email": "user@example.com", "password": "password123", "full_name": "John Doe" }
```

Response 201:
```json
{ "access_token": "eyJ...", "token_type": "bearer", "user": {...} }
```

### POST /api/v1/auth/login
Login. Same response shape as register.

### POST /api/v1/verify
Upload certificate for AI verification.

Request: multipart/form-data, field "file" (PDF/JPG/PNG, max 10 MB)

Response 201:
```json
{
  "id": "uuid",
  "verdict": "GENUINE",
  "trust_score": 82,
  "forgery_score": 0.12,
  "field_confidence": 0.91,
  "nlp_anomaly_score": 0.88,
  "institution_match": true,
  "institution_name": "IIT Bombay",
  "field_scores": {
    "STUDENT NAME": { "value": "Arjun Mehta", "confidence": 96 }
  },
  "nlp_reasoning": "ELA analysis shows uniform compression...",
  "issues": [],
  "heatmap_url": "/api/v1/heatmap/uuid",
  "report_url": "/api/v1/report/uuid",
  "processing_time_s": 4.21,
  "created_at": "2026-05-07 02:08"
}
```

Errors: 400 (bad file type), 413 (too large), 500 (ML failure)

### GET /api/v1/verify/{id}
Get a previous verification result.

### GET /api/v1/history?limit=50
Get user's verification history (most recent first).

### GET /api/v1/heatmap/{id}
Returns GradCAM heatmap as image/jpeg.

### GET /api/v1/report/{id}
Generates and returns a PDF forensic report (application/pdf).

### GET /api/v1/institutions?q=IIT
Search institution database.

### GET /health
```json
{ "status": "ok", "version": "2.5.0", "model_loaded": true }
```

---

## 10. Training the Model

### Dataset Layout

```
ml/training_data/
├── genuine/images/   1,008 real certificate images  (label = 1)
└── fake/images/      1,153 forged certificate images (label = 0)
```

### Run Training

```bash
cd certguard-ai/ml
python train.py
```

Best checkpoint saved to checkpoints/best_model.pth

### Current Results

| Metric | Value |
|---|---|
| Best epoch | 3 |
| Val AUC | 1.0000 |
| Test accuracy | 1.0000 |
| Test F1 | 1.0000 |

Note: Perfect scores indicate overfitting to the training distribution. More diverse data is needed for real-world use.

---

## 11. How the AI Pipeline Works

When a certificate is uploaded to POST /api/v1/verify:

```
Step 1  PDF Conversion (if needed)
        PDF first page rendered at 200 DPI to PNG

Step 2  Image Preprocessing
        Load → deskew → crop borders → CLAHE → resize → pad → normalise

Step 3  Error Level Analysis
        Re-save at JPEG quality 90 → pixel diff x10
        High-difference regions = likely tampered

Step 4  Forgery Detection (EfficientNet-B4)
        Input: 6-channel [RGB | ELA] tensor at 224x224
        Output: forgery_score + GradCAM heatmap + tamper regions

Step 5  OCR & Field Extraction
        Tesseract → regex → institution fuzzy match
        → roll number validation → sanity checks

Step 6  Trust Score Fusion
        genuine_conf = 1 - forgery_score
        nlp_score    = 1 - (warnings x 0.15)
        trust_score  = genuine_conf x 0.45
                     + field_conf   x 0.35
                     + nlp_score    x 0.20

Step 7  Verdict
        >= 75  GENUINE
        >= 45  SUSPICIOUS
        <  45  FAKE

Step 8  Save & Return
        GradCAM saved to heatmaps/{id}_heatmap.jpg
        Result stored in SQLite
        JSON response returned to frontend
```

---

## 12. Database Schema

### users
| Column | Type | Notes |
|---|---|---|
| id | String UUID | PK |
| email | String | Unique |
| full_name | String | |
| hashed_password | String | bcrypt |
| created_at | DateTime | UTC |

### verifications
| Column | Type | Notes |
|---|---|---|
| id | String UUID | PK = file_id |
| user_id | String | FK users.id |
| filename | String | |
| verdict | String | GENUINE/SUSPICIOUS/FAKE |
| trust_score | Integer | 0-100 |
| forgery_score | Float | 0.0-1.0 |
| field_confidence | Float | 0.0-1.0 |
| nlp_anomaly_score | Float | 0.0-1.0 |
| institution_match | Boolean | |
| institution_name | String | nullable |
| field_scores | Text | JSON string |
| nlp_reasoning | Text | |
| issues | Text | JSON array |
| heatmap_path | String | nullable |
| processing_time_s | Float | |
| created_at | DateTime | UTC |

### institutions
| Column | Type | Notes |
|---|---|---|
| id | Integer | Auto PK |
| name | String | Unique |
| short_name | String | nullable |
| location | String | nullable |
| verified | Boolean | default True |

---

## 13. Configuration & Environment

### Backend Environment Variables

| Variable | Default | Description |
|---|---|---|
| SECRET_KEY | CertiProof-secret-key-... | JWT signing key — change in production |
| ALLOWED_ORIGINS | http://localhost:5173,... | Comma-separated CORS origins |

### Model Configuration (ml/forgery_detector.py)

```python
MODEL_CONFIG = {
    "image_width":   224,    # increase for better accuracy
    "image_height":  224,
    "batch_size":    4,      # increase if GPU available
    "epochs":        15,
    "learning_rate": 3e-4,
    "dropout":       0.3,
    "model_name":    "efficientnet_b4",
    "num_classes":   2,
}
```

### Tesseract Auto-Detection (Windows)

Checked in order:
1. C:\Program Files\Tesseract-OCR\tesseract.exe
2. C:\Program Files (x86)\Tesseract-OCR\tesseract.exe
3. C:\Users\{USERNAME}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe

---

## 14. Known Limitations

**Model Overfitting**
The current checkpoint achieves 1.0 AUC because the fake images are all cropped from a small set of source documents. The model may not generalise to unseen certificate styles. More diverse training data is needed.

**OCR Accuracy**
Tesseract works best on clean, high-resolution printed text. Handwritten certificates, low-quality scans, or complex backgrounds may produce poor field extraction results.

**PDF Support**
pdf2image requires Poppler on Windows. Without it, PDF uploads fail.
Install from: https://github.com/oschwartz10612/poppler-windows

**No GPU**
Inference runs on CPU (~3-10 seconds per certificate). With a CUDA GPU this drops to under 1 second.

**LayoutLMv3 NER**
Loaded but uses base weights only. Not fine-tuned on certificate data. Disabled by default.

**Security**
- SECRET_KEY has a default value — must be changed before production
- No rate limiting on /api/v1/verify
- Uploaded files stored indefinitely — no cleanup job
