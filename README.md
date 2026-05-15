# 🛡 CertiProof — AI-Powered Certificate Authenticity Validator

![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?style=flat-square)
![Ollama](https://img.shields.io/badge/LLM-phi3%20via%20Ollama-8B5CF6?style=flat-square)

> AI-based forensic system to detect fake and manipulated academic certificates using deep learning, OCR, and local LLM reasoning.

---

## 📌 Problem Statement

The increasing number of fake and manipulated academic certificates has become a major concern for educational institutions, employers, and government agencies. There is no unified and reliable system to verify the authenticity of these documents efficiently.

---

## ✨ Features

- 🔬 **ELA Forensic Analysis** — Error Level Analysis detects tampered image regions
- 🧠 **EfficientNet-B4** — 6-channel deep learning model (RGB + ELA) trained on real/fake certificates
- 📋 **OCR Field Extraction** — Tesseract extracts student name, institution, degree, grade, roll number
- 🏛 **Institution Matching** — Fuzzy matching against 50 Indian universities with NAAC accreditation
- 🤖 **LLM Reasoning** — phi3/llama3 via Ollama generates forensic reasoning for each certificate
- 🗺 **GradCAM Heatmap** — Visual explanation of which regions influenced the verdict
- 💬 **AI Chatbot** — Context-aware assistant powered by phi3 for Q&A about results
- 📄 **PDF Report** — Downloadable forensic report with all findings
- 📧 **Email Report** — Send report to user's email via Gmail SMTP
- 🔐 **JWT Authentication** — Secure login/register with 7-day tokens

---

## 🏗 Architecture

```
Frontend (React + TypeScript)
        ↓ JWT Bearer
Backend (FastAPI + SQLite)
        ↓ Python imports
ML Layer (PyTorch)
  ├── EfficientNet-B4 + ELA  →  forgery_score + GradCAM
  ├── Tesseract OCR          →  extracted fields
  ├── Institution DB (50)    →  institution match
  └── Ollama (phi3)          →  AI reasoning
```

### Trust Score Formula
```
trust_score = (genuine_confidence × 0.45)
            + (field_confidence   × 0.35)
            + (nlp_score          × 0.20)

≥ 75  →  GENUINE
≥ 45  →  SUSPICIOUS
< 45  →  FAKE
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, Vite, jsPDF |
| Backend | FastAPI, SQLite, SQLAlchemy, Uvicorn |
| ML Model | EfficientNet-B4 (timm), PyTorch, OpenCV |
| OCR | Tesseract 5.x + pytesseract |
| LLM | phi3 / llama3 via Ollama |
| Auth | JWT (python-jose), bcrypt |
| PDF | ReportLab (server), jsPDF (client) |
| Email | Gmail SMTP (smtplib) |

---

## 📁 Project Structure

```
CertiProof/
├── src/                    # React frontend
│   ├── pages/Index.tsx     # Main SPA (all pages)
│   ├── lib/api.ts          # API client
│   └── components/
│       └── Chatbot.tsx     # AI chatbot widget
├── backend/
│   ├── main.py             # FastAPI app + ML pipeline
│   ├── requirements.txt    # Python dependencies
│   └── .env.example        # Environment template
└── ml/
    ├── forgery_detector.py # EfficientNet-B4 + ELA model
    ├── field_extractor.py  # OCR + field extraction
    ├── institution_list.py # 50 Indian universities
    ├── train.py            # Training pipeline
    └── checkpoints/        # Trained model weights
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (Windows)
- [Ollama](https://ollama.com) with phi3 or llama3

### 1. Install ML dependencies
```bash
cd ml
pip install torch torchvision timm opencv-python Pillow numpy scikit-learn mlflow tqdm pytesseract fuzzywuzzy python-Levenshtein ollama
```

### 2. Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
pip install python-dotenv
```

### 3. Install frontend dependencies
```bash
npm install
```

### 4. Configure environment
```bash
cd backend
copy .env.example .env
# Edit .env with your credentials
```

### 5. Run the application

**Terminal 1 — Ollama**
```bash
ollama serve
```

**Terminal 2 — Backend**
```bash
cd backend
python main.py
```

**Terminal 3 — Frontend**
```bash
npm run dev
```

Open: `http://localhost:8080`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/login` | Login and get JWT |
| POST | `/api/v1/verify` | Upload and analyze certificate |
| GET | `/api/v1/verify/{id}` | Get verification result |
| GET | `/api/v1/history` | User's verification history |
| GET | `/api/v1/heatmap/{id}` | GradCAM heatmap image |
| GET | `/api/v1/report/{id}` | Download PDF report |
| POST | `/api/v1/send-report/{id}` | Email report to user |
| POST | `/api/v1/chat` | AI chatbot (streaming) |
| GET | `/api/v1/institutions` | Search institution database |
| GET | `/health` | Health check |

Interactive docs: `http://localhost:8000/docs`

---

## 🧠 Model Details

| Component | Detail |
|---|---|
| Backbone | EfficientNet-B4 (pretrained ImageNet) |
| Input | 6 channels — 3 RGB + 3 ELA at 224x224 |
| Head | Dropout → Linear(1792,512) → GELU → Dropout → Linear(512,2) |
| Parameters | 18,468,954 trainable |
| Training data | 1,008 genuine + 1,153 fake certificates |
| Split | 70% train / 15% val / 15% test (stratified) |
| Optimizer | AdamW (lr=3e-4) + CosineAnnealingLR |

---

## 📧 Email Configuration

To enable the Send to Email feature, add to `backend/.env`:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-gmail@gmail.com
SMTP_PASSWORD=your-16-char-app-password
```

Get an App Password at: https://myaccount.google.com/apppasswords

---

## 🏛 Institution Database

50 Indian universities including IITs, NITs, DU, JNU, VIT, SRM, BITS, Amity, Manipal and more — each with NAAC accreditation grade and roll number regex pattern for validation.

---

## 📄 License

This project is for educational purposes.

---

## 👨‍💻 Author

**Shresth Kumar Verma** — AI-Based Academic Certificate Authenticity Validator
