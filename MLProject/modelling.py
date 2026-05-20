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

# ========== SETUP MLFLOW TRACKING (FILE-BASED) ==========
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CI_CD_Experiment")

print(f"✅ Tracking URI: file:./mlruns")

# ========== END ACTIVE RUN IF EXISTS ==========
if mlflow.active_run():
    print(f"⚠️ Found active run: {mlflow.active_run().info.run_id}")
    print("🏁 Ending active run...")
    mlflow.end_run()
    print("✅ Active run ended")

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

# ========== TRAINING WITH AUTOLOG ONLY ==========
model = RandomForestRegressor(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth if args.max_depth != 0 else None,
    random_state=args.random_state
)

# Start new run (pastikan tidak konflik)
with mlflow.start_run(run_name="CI_CD_Training", nested=False) as run:
    print(f"✅ Run started with ID: {run.info.run_id}")
    
    # ONLY autolog - NO manual logging
    mlflow.sklearn.autolog()
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Manual calculation for display only (not logged to MLflow)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 RESULTS:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.4f}")
    
    # Save run_id to file (for later steps)
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)
    print(f"✅ Run ID saved: {run.info.run_id}")

print("\n✅ MODELLING COMPLETE!")