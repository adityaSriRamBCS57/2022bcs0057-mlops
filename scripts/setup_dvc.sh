#!/bin/bash
# Run this ONCE locally after cloning the repo and setting up AWS credentials
# Usage: bash scripts/setup_dvc.sh <your-s3-bucket-name>

BUCKET=${1:-"2022bcs0057-mlops-bucket"}

echo "===== DVC + S3 Setup for 2022BCS0057 ====="

# 1. Create S3 bucket
echo "[1] Creating S3 bucket: $BUCKET"
aws s3 mb s3://$BUCKET --region us-east-1 || echo "Bucket may already exist, continuing..."

# 2. Init DVC
echo "[2] Initializing DVC"
dvc init --no-scm 2>/dev/null || dvc init

# 3. Set remote
echo "[3] Configuring S3 remote"
dvc remote add -d myS3 s3://$BUCKET/dvc-store -f

# 4. Generate datasets
echo "[4] Generating datasets via Python"
python3 -c "
from sklearn.datasets import load_iris
import pandas as pd, os
os.makedirs('data', exist_ok=True)
iris = load_iris()
import pandas as pd
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.iloc[:100].to_csv('data/iris_v1.csv', index=False)
df.to_csv('data/iris_v2.csv', index=False)
print('Datasets created.')
"

# 5. VERSION 1 — add v1, commit, push
echo "[5] Pushing Version 1 (partial dataset)"
cp data/iris_v1.csv data/iris.csv
dvc add data/iris.csv
git add data/iris.csv.dvc data/.gitignore .dvc/config
git commit -m "Dataset v1: partial iris (100 rows) [2022BCS0057]" 2>/dev/null || echo "git commit skipped"
dvc push

# 6. VERSION 2 — replace with full dataset
echo "[6] Pushing Version 2 (full dataset)"
cp data/iris_v2.csv data/iris.csv
dvc add data/iris.csv
git add data/iris.csv.dvc
git commit -m "Dataset v2: full iris (150 rows) [2022BCS0057]" 2>/dev/null || echo "git commit skipped"
dvc push

echo ""
echo "✅ DVC setup complete!"
echo "   Bucket : s3://$BUCKET/dvc-store"
echo "   v1     : 100 rows (first commit)"
echo "   v2     : 150 rows (second commit)"
echo ""
echo "Run 'git log --oneline' to see both commits."
