FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /code

# Install system dependencies efficiently
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (for better caching)
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt && \
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Copy application files
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /.cache .chroma /root/.cache/huggingface && \
    chmod 777 /.cache .chroma /root/.cache/huggingface

# Expose port
EXPOSE 7860

# Container-level health check against the app's /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health').status==200 else 1)" || exit 1

# Use exec form with optimized settings
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
