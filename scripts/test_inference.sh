#!/bin/bash
# Inference validation script — Part 8
# Usage: bash scripts/test_inference.sh

IMAGE="adityasr57/2022BCS0057-mlops:latest"

echo "===== Inference Validation - 2022BCS0057 ====="

echo "[1] Pulling Docker image..."
docker pull $IMAGE

echo "[2] Running container..."
docker run -d -p 8000:8000 --name iris_api $IMAGE
sleep 8

echo ""
echo "[3] Health check:"
curl -s http://localhost:8000/health | python3 -m json.tool

echo ""
echo "[4] Prediction test (all 4 features for setosa):"
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5]}' | python3 -m json.tool

echo ""
echo "[5] Prediction test (virginica-like):"
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [6.7, 5.8]}' | python3 -m json.tool

echo ""
echo "[6] Stopping container..."
docker stop iris_api && docker rm iris_api

echo ""
echo "✅ Inference validation complete!"
