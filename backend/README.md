# CertiProof Backend

FastAPI backend for AI-powered certificate verification.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Server runs on `http://localhost:8000`

## API Documentation

Interactive docs: `http://localhost:8000/docs`

## Endpoints

### Auth
- `POST /api/v1/auth/register` — Register new user
- `POST /api/v1/auth/login` — Login and get JWT token

### Verification
- `POST /api/v1/verify` — Upload certificate for verification (requires auth)
- `GET /api/v1/verify/{id}` — Get verification result by ID (requires auth)
- `GET /api/v1/history` — Get verification history (requires auth)

### Resources
- `GET /api/v1/heatmap/{id}` — Get GradCAM heatmap image
- `GET /api/v1/report/{id}` — Get PDF report
- `GET /api/v1/institutions?q=IIT` — Search institution database

### Health
- `GET /health` — Health check

## Database

SQLite database (`certiproof.db`) with tables:
- `users` — User accounts
- `verifications` — Verification results
- `institutions` — Verified institution database (50 Indian universities)

## AI Pipeline

1. Image preprocessing & ELA — Error Level Analysis for tampering detection
2. Forgery detection — EfficientNet-B4 (6-channel RGB + ELA)
3. OCR & field extraction — Tesseract + regex + fuzzy institution matching
4. Institution database lookup — 50 Indian universities
5. Trust score fusion — `(forgery*0.45) + (field*0.35) + (nlp*0.20)`

## Security Notes

- Change `SECRET_KEY` via environment variable in production
- Use HTTPS in production
- Add rate limiting for `/api/v1/verify` endpoint
