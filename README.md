# Titanic Survival Predictor

A machine learning web application that predicts whether a Titanic passenger would have survived, based on their attributes.

Built to demonstrate a complete ML engineering workflow: from data exploration and experiment tracking through to a deployed REST API and React frontend.

## Tech Stack

- **ML & Experiment Tracking:** scikit-learn, XGBoost, MLflow
- **API:** FastAPI, Pydantic, Pytest
- **Frontend:** TypeScript, React, Vite
- **Infrastructure:** Docker, Docker Compose

## Project Structure
```
titanic-predictor/
├── data/               # Dataset (see setup instructions)
├── notebooks/          # EDA and exploration
├── training/           # Model training and MLflow tracking
├── api/                # FastAPI backend
├── frontend/           # React + TypeScript frontend
└── tests/              # API tests
```

## Setup

### Prerequisites
- Python 3.14+
- Node.js 18+
- uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Installation

1. Clone the repo
```bash
   git clone <your-repo-url>
   cd titanic-predictor
```

2. Create and activate a virtual environment
```bash
   uv venv
   source venv/bin/activate  # Mac/Linux
```

3. Install dependencies
```bash
   uv pip install -r requirements.txt
```

4. Download the Titanic dataset from [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset) and save to `data/titanic.csv`

### Running the Project

**Train models and track experiments:**
```bash
cd training
python train.py
```

**View MLflow experiment dashboard:**
```bash
mlflow ui
# Open http://localhost:5000
```

**Start the API:**
```bash
cd api
uvicorn main:app --reload
# Open http://localhost:8000/docs
```

**Start the frontend:**
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

**Run with Docker:**
```bash
docker-compose up --build
```

## ML Approach

Three models were trained and compared using MLflow experiment tracking:

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 0.810 | 0.764 |
| Random Forest | 0.799 | 0.753 |
| XGBoost | 0.804 | 0.762 |

Logistic Regression was selected as the best model based on F1 score.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Predict survival probability |

## License
MIT
```