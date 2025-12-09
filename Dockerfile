# GraphLLM - Hugging Face Spaces Deployment
# Optimized Docker image for HF Spaces

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    API_PORT=7860 \
    HF_HOME=/app/cache \
    TRANSFORMERS_CACHE=/app/cache \
    SENTENCE_TRANSFORMERS_HOME=/app/cache

# Set working directory
WORKDIR /app

# Install system dependencies (minimal set for HF Spaces)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tesseract-ocr \
    ghostscript \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories with proper permissions (777 for HF Spaces non-root user)
RUN mkdir -p data uploads logs cache data/faiss_index && \
    chmod -R 777 data uploads logs cache

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
# HF Spaces expects the app to listen on 0.0.0.0:7860
CMD ["python3", "main.py"]
