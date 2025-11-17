# Script para cargar wisdom chunks en PostgreSQL
Write-Host "=== CARIA WISDOM INGESTION ===" -ForegroundColor Cyan

# Verificar que PostgreSQL esta corriendo
Write-Host "`nVerificando PostgreSQL..." -ForegroundColor Yellow
try {
    docker ps --filter "name=caria" --format "table {{.Names}}\t{{.Status}}"
} catch {
    Write-Host "ERROR: Docker no esta corriendo" -ForegroundColor Red
    exit 1
}

# Ejecutar ingestion
Write-Host "`nIniciando ingestion de wisdom chunks..." -ForegroundColor Green
cd caria_data
poetry run python scripts/ingestion/ingest_wisdom.py

Write-Host "`nCompletado!" -ForegroundColor Green
