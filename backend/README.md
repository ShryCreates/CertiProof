# CertValidator Backend

FastAPI backend for AI-powered certificate verification.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
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
- `GET /api/v1/heatmap/{id}` — Get heatmap image (not yet implemented)
- `GET /api/v1/report/{id}` — Get PDF report (not yet implemented)
- `GET /api/v1/institutions?q=IIT` — Search institution database

### Health
- `GET /health` — Health check

## Database

SQLite database (`certvalidator.db`) with tables:
- `users` — User accounts
- `verifications` — Verification results
- `institutions` — Verified institution database (seeded with 10 Indian universities)

## Mock ML Pipeline

The current implementation uses a mock ML pipeline that generates random scores. In production, replace `mock_ml_pipeline()` with actual ML models:

1. **Image preprocessing & ELA** — Error Level Analysis for tampering detection
2. **Forgery detection** — EfficientNet-B4 or similar CNN
3. **OCR & field extraction** — LayoutLMv3 or Tesseract + regex
4. **Institution database lookup** — Query verified institutions
5. **LLM reasoning** — Mistral-7B or GPT for anomaly explanation
6. **Trust score fusion** — Weighted combination: `(forgery*0.45) + (field*0.35) + (nlp*0.20)`

## Security Notes

- Change `SECRET_KEY` in production
- Use HTTPS in production
- Add rate limiting for `/api/v1/verify` endpoint
- Implement file size limits (currently unlimited)
- Add virus scanning for uploaded files
- Use environment variables for secrets
