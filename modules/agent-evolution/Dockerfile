FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-phase2.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-phase2.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY api/ ./api/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/static

# Copy static files for web interface
COPY src/web/static/ /app/static/

# Environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV DEAN_MODULE_PATH=/app

# Expose ports
EXPOSE 8090 8091

# Default command - can be overridden
CMD ["python", "-m", "src.web.control_interface"]