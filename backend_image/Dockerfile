# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Avoid Python buffering issues (important for logs in Cloud Run)
ENV PYTHONUNBUFFERED=1

# Install system dependencies (important for xgboost, numpy, sklearn)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --default-timeout=1000 -r requirements.txt

# Copy ONLY backend-required code
COPY api/ api/
COPY candidate_generation/ candidate_generation/
COPY data ./data
COPY models/ models/
COPY pipeline/ pipeline/
COPY ranking/ ranking/

# Expose FastAPI port
EXPOSE 8080

# Start FastAPI server
CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
