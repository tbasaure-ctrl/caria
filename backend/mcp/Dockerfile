FROM python:3.11-slim

WORKDIR /app

COPY ../../pyproject.toml ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root --only main

COPY ../../src ./src
COPY ./server.py ./server.py

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "services.mcp.server:app", "--host", "0.0.0.0", "--port", "8000"]

