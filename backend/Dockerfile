# SyncNet FastAPI Backend Microservice Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY backend/requirements.txt /app/backend/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application files, data features, models, and docs
COPY backend /app/backend
COPY data /app/data
COPY models /app/models
COPY docs /app/docs

# Expose API port
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app

# Run FastAPI app via Uvicorn server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
