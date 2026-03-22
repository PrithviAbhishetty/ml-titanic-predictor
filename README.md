# ml-titanic-predictor

![CI](https://github.com/PrithviAbhishetty/ml-titanic-predictor/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.14-blue)
![CV F1](https://img.shields.io/badge/cv--f1-0.754-brightgreen)

End-to-end ML portfolio project — Titanic survival predictor with a scikit-learn/XGBoost model, MLflow experiment tracking, FastAPI backend, and React/TypeScript frontend. Built to demonstrate the full ML engineering lifecycle: from model training and experiment tracking through containerised deployment, CI/CD, and a layered test suite.

**Live:** [Frontend](https://ml-titanic-predictor.vercel.app) · [Backend](https://ml-titanic-predictor.onrender.com) · [MLflow on DagsHub](https://dagshub.com/PrithviAbhishetty/ml-titanic-predictor)

---

## Architecture

```
User Browser
    │
    ▼
Vercel (React + TypeScript frontend)
    │  VITE_API_URL
    ▼
Render (FastAPI backend)
    │  loads model at startup via DVC
    ▼
DagsHub (model storage + MLflow tracking server)
```

The model is not committed to git. It is tracked by DVC with DagsHub as the remote, pulled at build time on Render via `dvc pull`. MLflow experiment runs are logged to the DagsHub-hosted tracking server during training.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| ML | scikit-learn, XGBoost | Industry-standard; XGBoost consistently outperformed LR and RF on CV F1 |
| Experiment tracking | MLflow + DagsHub | Remote tracking server without self-hosting infrastructure |
| Model versioning | DVC | Keeps large binaries out of git while maintaining reproducibility |
| Backend | FastAPI | Async, type-safe, automatic OpenAPI docs, production-grade |
| Frontend | React + TypeScript + Vite | Type safety end-to-end; Vite for fast local dev |
| CI/CD | GitHub Actions | Native GitHub integration; matrix of unit → integration → E2E on PRs |
| Hosting | Render + Vercel | Free tier sufficient; Vercel preview deployments enable E2E tests on PRs |
| Monitoring | UptimeRobot | Prevents Render free tier spin-down via `/health` pings every 5 min |

---

## Project Structure

```
ml-titanic-predictor/
├── api/                   # FastAPI application
├── frontend/              # React + TypeScript + Vite
├── training/              # ML pipeline and MLflow tracking
├── tests/                 # Unit, integration, and E2E tests
├── models/                # DVC pointer (model stored on DagsHub)
├── .github/workflows/     # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

Each subdirectory has its own README with further detail.

---

## Local Setup

### Prerequisites
- Python 3.14
- [uv](https://github.com/astral-sh/uv)
- Node.js / npm
- DagsHub account with access to the repository (for `dvc pull`)

### Install

```bash
git clone https://github.com/PrithviAbhishetty/ml-titanic-predictor
cd ml-titanic-predictor
uv sync --extra test
dvc pull  # downloads best_model.joblib from DagsHub
```

### Run backend

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run frontend

```bash
# Create frontend/.env.local with:
# VITE_API_URL=http://localhost:8000
npm run dev --prefix frontend
```

### Run tests

```bash
uv run pytest tests/test_api.py -v           # unit
uv run pytest tests/test_integration.py -v  # integration
uv run pytest tests/test_e2e.py -v          # E2E (requires frontend + backend running)
```

### Retrain model

```bash
DAGSHUB_USER_TOKEN=<your_token> uv run python training/train.py
dvc push
```

---

## CI/CD Pipeline

Every push and PR to `main` runs: install → train (cached) → validate (F1 > 0.7) → unit tests → integration tests.

On PRs additionally: deploy to staging Render service → resolve Vercel preview URL → run Playwright E2E tests against the live preview.

On merge to `main`: Render and Vercel auto-deploy production.

See [`.github/workflows/test.yml`](.github/workflows/test.yml) for full pipeline definition and [`tests/README.md`](tests/README.md) for test suite detail.

---

## Design Decisions

**Why not commit the model to git?** Model files are binary, potentially large, and change independently of application code. DVC gives the model its own versioning lifecycle — a commit to the DVC pointer file (`.dvc`) records exactly which model artifact corresponds to which code state, without bloating the git history.

**Why MLflow on a portfolio project?** MLflow's value here is partly operational (comparing LR, RF, and XGBoost runs in a UI) and partly demonstrative — `mlflow.sklearn.log_model()` is the industry pattern for logging and later loading models in serving code. The FastAPI `ModelService` loads directly from the MLflow model registry, which reflects how this would work in a real system.

**Why a three-layer test suite?** Unit tests (TestClient) run in milliseconds and catch regressions. Integration tests (real uvicorn subprocess) verify CORS, headers, and response times — things TestClient abstracts away. E2E tests (Playwright) verify the frontend→backend contract on real deployed infrastructure. Each layer catches a different class of failure and the cost/speed tradeoff is intentional.