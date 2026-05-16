"""
modelling.py - Training model untuk CI/CD
Dengan penanganan active run conflict
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
print("MLflow Project - Training Model (with Conda)")
print("="*60)

# ========== PARSE ARGUMENTS ==========
parser = argparse.ArgumentParser()
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

print(f"📌 Parameters: test_size={args.test_size}, random_state={args.random_state}")

# ========== SETUP MLFLOW ==========
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CI_CD_Experiment")

print(f"✅ Tracking URI: file:./mlruns")
print(f"✅ MLflow version: {mlflow.__version__}")

# ========== END ANY ACTIVE RUN ==========
print("\n🔄 Checking for active runs...")
if mlflow.active_run():
    print(f"⚠️ Found active run: {mlflow.active_run().info.run_id}")
    print("🏁 Ending active run...")
    mlflow.end_run()
    print("✅ Active run ended")
else:
    print("✅ No active run found")

# ========== LOAD DATA ==========
print("\n📊 Loading dataset...")
df = pd.read_csv('customer_shopping_data_preprocessing.csv')
print(f"✅ Dataset loaded: {df.shape}")

# ========== PREPARE DATA ==========
print("\n🔧 Preparing data...")
target = 'total_amount'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

print(f"✅ Train: {len(X_train)} samples")
print(f"✅ Test: {len(X_test)} samples")

# ========== TRAINING ==========
print("\n🚀 Starting training...")
model = RandomForestRegressor(n_estimators=100, random_state=args.random_state)

# Start new run (dipastikan tidak ada konflik)
with mlflow.start_run(run_name="CI_CD_Training") as run:
    print(f"✅ Run started with ID: {run.info.run_id}")
    
    # Log parameters
    mlflow.log_params({
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_estimators": 100,
        "model_type": "RandomForestRegressor"
    })
    print("✅ Parameters logged")
    
    # Train
    print("🔄 Training RandomForest...")
    model.fit(X_train, y_train)
    print("✅ Training completed")
    
    # Predict
    print("🔄 Making predictions...")
    y_pred = model.predict(X_test)
    print("✅ Predictions completed")
    
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
    print("✅ Metrics logged")
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    print("✅ Model logged")
    
    # Save run_id
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)
    
    print(f"\n📊 RESULTS:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE : {mae:.4f}")
    print(f"   R²  : {r2:.4f}")
    print(f"\n✅ Run ID: {run.info.run_id}")

print("\n🎉 MODELLING COMPLETE!")