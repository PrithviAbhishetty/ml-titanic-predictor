# api/

FastAPI backend for the Titanic survival predictor. Exposes a `/predict` endpoint that runs inference against a trained XGBoost model loaded from the MLflow model registry via DVC.

---

## Structure

```
api/
├── main.py       # FastAPI app, CORS config, rate limiting, endpoint definitions
├── model.py      # ModelService — loads and caches the model at startup
├── schemas.py    # Pydantic request/response schemas
└── __init__.py
```

---

## Endpoints

### `GET /health`
Liveness check. Returns `{"status": "ok"}`. Accepts both GET and HEAD (HEAD used by UptimeRobot).

### `POST /predict`
Run survival prediction for a single passenger.

**Request body:**
```json
{
  "pclass": 1,
  "sex": "female",
  "age": 29,
  "sibsp": 0,
  "parch": 0,
  "fare": 100.0,
  "embarked": "S"
}
```

**Response:**
```json
{
  "survived": true,
  "survival_probability": 0.94
}
```

Rate limited to 10 requests/minute per IP via `slowapi`.

---

## Model Loading

`ModelService` in `model.py` loads `best_model.joblib` at application startup. The model file is not in git — it is pulled from DagsHub via DVC during the Render build step. On startup failure (model file missing), the app will raise immediately rather than serve requests against an uninitialised model.

---

## CORS

Configured to allow:
- `https://ml-titanic-predictor*.vercel.app` (production + preview deployments)
- `http://localhost:5173` (local frontend dev)

Methods: `GET`, `POST`, `OPTIONS`, `HEAD`.

---

## Running Locally

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Requires `best_model.joblib` to be present in `models/`. Run `dvc pull` first if it is missing.

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DAGSHUB_USER_TOKEN` | Authentication for DVC pull at build time |
| `DVC_SITE_CACHE_DIR` | Set to `/tmp/dvc` on Render (free tier `/var/tmp` is read-only) |