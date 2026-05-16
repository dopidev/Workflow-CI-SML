# Dockerfile untuk MLflow model dengan Python 3.12
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MLflow
RUN pip install mlflow==2.19.0

# Copy model
COPY mlruns ./mlruns

# Set environment variables
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

# Expose port for MLserver
EXPOSE 8080

# Run the model server
ENTRYPOINT ["mlflow", "models", "serve", "-m", "random_forest_model", "-h", "0.0.0.0", "-p", "8080"]