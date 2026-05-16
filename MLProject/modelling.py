"""
modelling.py - MLflow Project untuk Kriteria 3 (Advanced)
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

# ========== BACA PARAMETER ==========
parser = argparse.ArgumentParser()
parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
parser.add_argument('--random_state', type=int, default=42, help='Random state')
parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
parser.add_argument('--max_depth', type=int, default=10, help='Max depth of trees')
args = parser.parse_args()

print("="*60)
print("MLflow Project - Training Model (Advanced)")
print("="*60)

print(f"\n📌 Parameters used:")
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
print("\n2️⃣ Loading dataset...")

df = pd.read_csv('customer_shopping_data_preprocessing.csv')
print(f"✅ Dataset loaded! Shape: {df.shape}")

# ========== PISAHKAN FITUR DAN TARGET ==========
print("\n3️⃣ Separating features (X) and target (y)...")

target = 'total_amount'
X = df.drop(columns=[target])
y = df[target]

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")

# ========== SPLIT DATA ==========
print("\n4️⃣ Splitting data into train and test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

print(f"✅ Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"✅ Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# ========== TRAINING ==========
print("\n5️⃣ Training RandomForestRegressor model...")

model = RandomForestRegressor(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    random_state=args.random_state
)

with mlflow.start_run(run_name="CI_CD_Training_Advanced") as run:
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
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metrics({
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    })
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Save run_id for artifacts
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE : {mae:.4f}")
    print(f"   R²  : {r2:.4f}")
    
    print(f"\n✅ Run ID: {run_id}")

print("\n✅ MODELLING COMPLETED SUCCESSFULLY!")