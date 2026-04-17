#!/bin/bash
# Run this locally to train and view MLflow UI
# Usage: bash scripts/run_local.sh

echo "===== Local Training - 2022BCS0057 ====="

pip install -r requirements.txt -q

# Start MLflow in background
echo "[1] Starting MLflow server at http://localhost:5000"
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns &
MLFLOW_PID=$!
sleep 4

# Run training
echo "[2] Running training with 5 experiments..."
MLFLOW_TRACKING_URI=http://localhost:5000 python train.py

echo ""
echo "✅ Training done!"
echo "   Open http://localhost:5000 to view MLflow UI"
echo "   Experiment: 2022BCS0057_experiment"
echo ""
echo "Press Ctrl+C to stop MLflow server."
wait $MLFLOW_PID
