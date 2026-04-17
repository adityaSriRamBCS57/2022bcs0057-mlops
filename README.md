# 2022BCS0057-mlops

**Student:** Aditya Sri Ram  
**Roll No:** 2022BCS0057  
**MLflow Experiment:** `2022BCS0057_experiment`  
**Docker Image:** `adityasr57/2022BCS0057-mlops`

---

## Repository Structure

```
2022BCS0057-mlops/
├── train.py                        # MLflow training (5 runs)
├── app.py                          # FastAPI inference API
├── Dockerfile                      # Docker build
├── requirements.txt
├── data/                           # DVC-tracked datasets
│   ├── iris_v1.csv                 # Version 1 (100 rows)
│   └── iris_v2.csv                 # Version 2 (150 rows)
├── models/                         # Saved model artifacts
├── .dvc/config                     # DVC + S3 config
├── .github/workflows/mlops.yml     # GitHub Actions CI/CD
└── scripts/
    ├── setup_dvc.sh                # One-time DVC + S3 setup
    ├── run_local.sh                # Local training + MLflow
    └── test_inference.sh           # Docker inference test
```

---

## Quick Setup

### Step 1 — Clone and install
```bash
git clone https://github.com/<your-username>/2022BCS0057-mlops.git
cd 2022BCS0057-mlops
pip install -r requirements.txt
```

### Step 2 — Configure AWS (from AWS Academy)
```bash
export AWS_ACCESS_KEY_ID=<from-academy>
export AWS_SECRET_ACCESS_KEY=<from-academy>
export AWS_SESSION_TOKEN=<from-academy>
```

### Step 3 — DVC + S3 setup (run once)
```bash
bash scripts/setup_dvc.sh 2022bcs0057-mlops-bucket
```

### Step 4 — Run training locally
```bash
bash scripts/run_local.sh
# Opens MLflow at http://localhost:5000
```

### Step 5 — Build and push Docker image
```bash
docker build -t adityasr57/2022BCS0057-mlops:latest .
docker push adityasr57/2022BCS0057-mlops:latest
```

### Step 6 — Inference validation
```bash
bash scripts/test_inference.sh
```

---

## GitHub Actions Secrets Required

| Secret | Value |
|--------|-------|
| `AWS_ACCESS_KEY_ID` | From AWS Academy |
| `AWS_SECRET_ACCESS_KEY` | From AWS Academy |
| `AWS_SESSION_TOKEN` | From AWS Academy |
| `DOCKERHUB_USERNAME` | `adityasr57` |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns name + roll no |
| `/predict` | POST | Returns prediction + name + roll no |

### Sample predict request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5]}'
```
Features order: `sepal length (cm)`, `petal length (cm)` (reduced feature set from Run 4 & 5)
