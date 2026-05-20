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

print("="*60)
print("MLflow Project - Training Model (Kriteria 3)")
print("="*60)

# ========== PARSE ARGUMENTS ==========
parser = argparse.ArgumentParser()
parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
parser.add_argument('--random_state', type=int, default=42, help='Random state')
parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
parser.add_argument('--max_depth', type=int, default=10, help='Max depth')
args = parser.parse_args()

print(f"📌 Parameters: test_size={args.test_size}, random_state={args.random_state}, n_estimators={args.n_estimators}, max_depth={args.max_depth}")

# ========== SETUP MLFLOW TRACKING ==========
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CI_CD_Experiment")

print(f"✅ Tracking URI: file:./mlruns")
print(f"✅ Experiment: CI_CD_Experiment")

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

# Enable autolog (will log metrics, params, etc.)
mlflow.sklearn.autolog()

# Train model
model.fit(X_train, y_train)

# ========== MANUAL LOG MODEL (TANPA start_run) ==========
# Karena MLflow Project sudah memiliki run aktif, kita bisa langsung log model
# Ini penting agar model tersimpan di artifacts untuk Docker build
mlflow.sklearn.log_model(model, "random_forest_model")

# Predict for display
y_pred = model.predict(X_test)

# Calculate metrics (for display only)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 RESULTS:")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE: {mae:.4f}")
print(f"   R²: {r2:.4f}")

# Get current run ID
current_run = mlflow.last_active_run()
if current_run:
    run_id = current_run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    print(f"✅ Run ID saved: {run_id}")
    print(f"✅ Model saved as artifact: random_forest_model")

print("\n✅ MODELLING COMPLETE!")