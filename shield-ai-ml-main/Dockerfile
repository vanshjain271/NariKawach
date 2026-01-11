FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 shielduser && chown -R shielduser:shielduser /app
USER shielduser

# Create necessary directories
RUN mkdir -p storage/models storage/logs storage/cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]