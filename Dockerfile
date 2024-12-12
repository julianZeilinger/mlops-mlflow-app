FROM python:3.8-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*
# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC
RUN pip install --no-cache-dir dvc[s3]

# Copy the application code
COPY . /app
WORKDIR /app

# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=http://mlflow-tracking.mlflow.svc.cluster.local:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://mlflow-minio.mlflow.svc.cluster.local:80