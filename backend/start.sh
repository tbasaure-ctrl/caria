#!/bin/bash
# Startup script that sets PYTHONPATH before starting uvicorn
# This ensures Python can find the caria module

# Set PYTHONPATH before any Python execution
export PYTHONPATH=/app/caria-lib:/app/backend:$PYTHONPATH

# Print PYTHONPATH for debugging
echo "PYTHONPATH: $PYTHONPATH"
echo "Checking if caria-lib exists..."
ls -la /app/caria-lib/ || echo "Warning: /app/caria-lib/ not found"
echo "Checking if caria/models exists..."
ls -la /app/caria-lib/caria/models/ || echo "Warning: /app/caria-lib/caria/models/ not found"
echo "Verifying caria module can be imported..."
python3 -c "import sys; sys.path.insert(0, '/app/caria-lib'); print('sys.path:', sys.path); import caria; print('✓ caria imported'); import caria.models; print('✓ caria.models imported'); import caria.models.auth; print('✓ caria.models.auth imported successfully')" || echo "✗ Warning: Could not import caria.models.auth"

# Change to backend directory to ensure correct module resolution
cd /app/backend

# Use python -m uvicorn instead of uvicorn directly
# This ensures PYTHONPATH is respected when Python imports modules
# The PYTHONPATH must be set before Python starts, which is why we use export above
# Cloud Run sets PORT automatically (usually 8080)
# Railway and other platforms also use PORT env var
# Cloud Run sets PORT=8080 automatically
PORT=${PORT:-8080}
echo "Starting server on port $PORT"
echo "PYTHONPATH: $PYTHONPATH"
echo "Environment variables:"
env | grep -E "(PORT|PYTHON|DATABASE)" || true
exec python3 -m uvicorn api.app:socketio_app --host 0.0.0.0 --port $PORT

