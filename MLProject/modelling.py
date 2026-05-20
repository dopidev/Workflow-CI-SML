"""
modelling.py - MLflow Project untuk Kriteria 3
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import argparse
import os
import joblib

print("="*60)
print("MLflow Project - Training Model (Kriteria 3)")
print("="*60)

# ========== PARSE ARGUMENTS ==========
parser = argparse.ArgumentParser()
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=42)
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=10)
args = parser.parse_args()

print(f"📌 Parameters: test_size={args.test_size}, random_state={args.random_state}, n_estimators={args.n_estimators}, max_depth={args.max_depth}")

# ========== SETUP ==========
os.makedirs("mlruns", exist_ok=True)
os.makedirs("model_artifacts", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CI_CD_Experiment")

print(f"✅ Tracking URI: file:./mlruns")

# ========== LOAD DATA ==========
df = pd.read_csv('customer_shopping_data_preprocessing.csv')
print(f"✅ Dataset loaded: {df.shape}")

# ========== PREPARE DATA ==========
target = 'total_amount'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

print(f"✅ Train: {len(X_train)} samples, Test: {len(X_test)} samples")

# ========== TRAINING ==========
model = RandomForestRegressor(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth if args.max_depth != 0 else None,
    random_state=args.random_state
)

# ========== TRAINING WITH MLFLOW ==========
with mlflow.start_run(run_name="CI_CD_Training") as run:
    run_id = run.info.run_id
    print(f"✅ Run ID: {run_id}")
    
    # Log parameters
    mlflow.log_params({
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth
    })
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metrics({
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    })
    
    # Log model (INI PENTING untuk Docker build)
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"\n📊 RESULTS:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.4f}")
    
    # Save run_id to file
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    print(f"✅ Run ID saved: {run_id}")
    
    # Also save model as pickle directly (fallback untuk Docker)
    joblib.dump(model, "model_artifacts/model.pkl")
    print("✅ Model also saved as model_artifacts/model.pkl")

print("\n✅ MODELLING COMPLETE!")