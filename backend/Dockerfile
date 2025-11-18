FROM python:3.11-slim

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
COPY backend/requirements.txt /tmp/api_requirements.txt
RUN pip install --no-cache-dir -r /tmp/api_requirements.txt && rm -f /tmp/api_requirements.txt

# Install caria-lib requirements
COPY caria-lib/requirements.txt /tmp/caria_lib_requirements.txt
RUN pip install --no-cache-dir -r /tmp/caria_lib_requirements.txt && rm -f /tmp/caria_lib_requirements.txt

# Copy EVERYTHING - no exclusions, no filtering
COPY backend/ /app/backend/
COPY caria-lib/ /app/caria-lib/
COPY configs/ /app/configs/
COPY caria_data/infrastructure/migrations/ /app/caria_data/infrastructure/migrations/
COPY caria_data/migrations/init.sql /app/caria_data/migrations/init.sql

# Copy models directory (may only contain .gitkeep if .pkl files are gitignored)
COPY caria_data/models/ /app/caria_data/models/

# Create models directory for runtime
RUN mkdir -p /app/models

# Copy regime model if it exists, otherwise warn (model may be gitignored)
RUN if [ -f /app/caria_data/models/regime_hmm_model.pkl ]; then \
        cp /app/caria_data/models/regime_hmm_model.pkl /app/models/regime_hmm_model.pkl && \
        echo "✓ Regime HMM model copied to /app/models/"; \
    else \
        echo "⚠ Warning: regime_hmm_model.pkl not found - regime detection may use defaults"; \
    fi

# Verify models directory exists
RUN ls -la /app/caria-lib/caria/models/ && \
    test -f /app/caria-lib/caria/models/auth.py && \
    echo "✓ models directory and auth.py verified" || \
    (echo "✗ ERROR: models directory missing!" && exit 1)

# Copy startup script
COPY backend/start.sh /app/backend/start.sh
RUN chmod +x /app/backend/start.sh

# Set PYTHONPATH
ENV PYTHONPATH=/app/caria-lib:/app/backend:$PYTHONPATH

WORKDIR /app/backend

EXPOSE 8000

CMD ["/app/backend/start.sh"]

