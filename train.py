import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

STUDENT_NAME = "Aditya Sri Ram"
ROLL_NO = "2022BCS0057"
EXPERIMENT_NAME = "2022BCS0057_experiment"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(EXPERIMENT_NAME)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Save full dataset (version 2)
os.makedirs("data", exist_ok=True)
df.to_csv("data/iris_v2.csv", index=False)

# Save partial dataset (version 1) - first 100 rows
df.iloc[:100].to_csv("data/iris_v1.csv", index=False)

os.makedirs("models", exist_ok=True)

def run_experiment(run_name, data_version, model_type, params, feature_cols):
    dataset_path = f"data/iris_v{data_version}.csv"
    data = pd.read_csv(dataset_path)

    X = data[feature_cols]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_type == "LogisticRegression":
        model = LogisticRegression(**params, max_iter=1000)
    else:
        model = RandomForestClassifier(**params, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("student_name", STUDENT_NAME)
        mlflow.log_param("roll_no", ROLL_NO)
        mlflow.log_param("dataset_version", f"v{data_version}")
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("features_used", str(feature_cols))
        mlflow.log_param("num_features", len(feature_cols))
        for k, v in params.items():
            mlflow.log_param(k, v)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        # Save metrics JSON
        metrics = {
            "student_name": STUDENT_NAME,
            "roll_no": ROLL_NO,
            "run_name": run_name,
            "dataset_version": f"v{data_version}",
            "model_type": model_type,
            "features_used": feature_cols,
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
        }
        metrics_path = f"models/metrics_{run_name.replace(' ', '_')}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[{run_name}] Accuracy: {acc:.4f} | F1: {f1:.4f}")

    return model, scaler, acc

ALL_FEATURES = list(iris.feature_names)
REDUCED_FEATURES = ["sepal length (cm)", "petal length (cm)"]

# Run 1: v1, LR, base config, all features
run_experiment("Run1_v1_LR_base", 1, "LogisticRegression",
               {"C": 1.0, "solver": "lbfgs"}, ALL_FEATURES)

# Run 2: v1, LR, hyperparameter change
run_experiment("Run2_v1_LR_tuned", 1, "LogisticRegression",
               {"C": 0.1, "solver": "lbfgs"}, ALL_FEATURES)

# Run 3: v2, LR, base config, all features
run_experiment("Run3_v2_LR_base", 2, "LogisticRegression",
               {"C": 1.0, "solver": "lbfgs"}, ALL_FEATURES)

# Run 4: v2, LR, feature selection (reduced features)
run_experiment("Run4_v2_LR_feat_sel", 2, "LogisticRegression",
               {"C": 1.0, "solver": "lbfgs"}, REDUCED_FEATURES)

# Run 5: v2, RandomForest, different model + feature selection
best_model, best_scaler, _ = run_experiment(
    "Run5_v2_RF_feat_sel", 2, "RandomForestClassifier",
    {"n_estimators": 100, "max_depth": 5}, REDUCED_FEATURES
)

# Save best model for inference
import joblib
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(best_scaler, "models/best_scaler.pkl")

feature_meta = {"features": REDUCED_FEATURES, "model_type": "RandomForestClassifier"}
with open("models/feature_meta.json", "w") as f:
    json.dump(feature_meta, f)

print("\n✅ All 5 runs complete. Best model saved.")
