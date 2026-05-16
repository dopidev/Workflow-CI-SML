"""
modelling.py - MLflow Project untuk Kriteria 3
Training model otomatis via GitHub Actions
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

# ========== BACA PARAMETER DARI COMMAND LINE ==========
parser = argparse.ArgumentParser()
parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
parser.add_argument('--random_state', type=int, default=42, help='Random state')
parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
parser.add_argument('--max_depth', type=int, default=10, help='Max depth of trees')
args = parser.parse_args()

print("="*60)
print("MLflow Project - Training Model")
print("="*60)

print(f"\n📌 Parameter yang digunakan:")
print(f"   test_size   : {args.test_size}")
print(f"   random_state: {args.random_state}")
print(f"   n_estimators: {args.n_estimators}")
print(f"   max_depth   : {args.max_depth}")

# ========== SETUP MLFLOW ==========
print("\n1️⃣ Setup MLflow tracking...")

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CI_CD_Experiment")

print(f"✅ Tracking URI: file:./mlruns")
print(f"✅ Experiment: CI_CD_Experiment")

# ========== LOAD DATA ==========
print("\n2️⃣ Memuat dataset...")

df = pd.read_csv('customer_shopping_data_preprocessing.csv')
print(f"✅ Dataset loaded! Shape: {df.shape}")

# ========== PISAHKAN FITUR DAN TARGET ==========
print("\n3️⃣ Pisahkan fitur (X) dan target (y)...")

target = 'total_amount'
X = df.drop(columns=[target])
y = df[target]

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")

# ========== SPLIT DATA ==========
print("\n4️⃣ Split data training dan testing...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

print(f"✅ Training set: {len(X_train)} samples")
print(f"✅ Testing set: {len(X_test)} samples")

# ========== TRAINING ==========
print("\n5️⃣ Training model RandomForestRegressor...")

model = RandomForestRegressor(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    random_state=args.random_state
)

with mlflow.start_run(run_name="CI_CD_Training") as run:
    # Simpan parameter
    mlflow.log_params({
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth
    })
    
    # Latih model
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung metrik
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Simpan metrik
    mlflow.log_metrics({
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    })
    
    # Simpan model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"\n📊 HASIL EVALUASI:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE : {mae:.4f}")
    print(f"   R²  : {r2:.4f}")

print("\n✅ MODELLING SELESAI!")