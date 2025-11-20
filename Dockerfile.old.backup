# Dockerfile para Caria API Multi-User
# Build context: notebooks/ directory (root of repository)
# This Dockerfile is optimized for Render deployment

FROM python:3.11-slim

# Build-time configuration
ENV PIP_DEFAULT_TIMEOUT=1800 \
    PIP_RETRIES=5 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install API requirements
COPY services/requirements.txt /tmp/api_requirements.txt
RUN pip install --no-cache-dir -r /tmp/api_requirements.txt \
    && rm -f /tmp/api_requirements.txt

# Install caria_data requirements
COPY caria_data/requirements.txt /tmp/caria_data_requirements.txt
RUN pip install --no-cache-dir -r /tmp/caria_data_requirements.txt \
    && rm -f /tmp/caria_data_requirements.txt

# Copy application source
COPY services/ /app/services/
COPY caria_data/ /app/caria_data/

# Copy and make startup script executable
COPY services/start.sh /app/services/start.sh
RUN chmod +x /app/services/start.sh

# Set PYTHONPATH to include both caria_data/src and services directory
# This ensures both 'caria' and 'api' modules can be imported
ENV PYTHONPATH=/app/caria_data/src:/app/services:$PYTHONPATH

# Verify the structure is correct and Python can import modules
RUN echo "PYTHONPATH: $PYTHONPATH" && \
    ls -la /app/caria_data/src/caria/models/ || echo "Warning: caria.models not found" && \
    ls -la /app/services/api/ || echo "Warning: api not found" && \
    python3 -c "import sys; sys.path.insert(0, '/app/caria_data/src'); import caria.models.auth; print('✓ caria.models.auth imported successfully')" || echo "✗ Failed to import caria.models.auth"

# Set working directory to services for the API
WORKDIR /app/services

# Expose port
EXPOSE 8000

# Healthcheck (using curl for better reliability)
# Railway sets PORT automatically, but healthcheck runs before env vars are available
# So we check both common ports
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || curl -f http://localhost:$PORT/health/live || exit 1

# Run application with SocketIO support per audit document
# Use startup script that sets PYTHONPATH before importing
# This ensures Python can find caria module before any imports happen
CMD ["/app/services/start.sh"]
