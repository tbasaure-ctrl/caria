# Script para verificar estado de la API
Write-Host "Verificando estado de Caria API..." -ForegroundColor Cyan

# Verificar si el container esta corriendo
$container = docker ps --filter "name=caria_api" --format "{{.Status}}"
Write-Host "Container status: $container" -ForegroundColor Yellow

# Intentar curl
Write-Host "`nIntentando conectar a API..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2
    Write-Host "API ESTA LISTA!" -ForegroundColor Green
    $response.Content
} catch {
    Write-Host "API aun no esta lista. Instalando dependencias..." -ForegroundColor Yellow
    Write-Host "Ver logs con: docker logs -f caria_api" -ForegroundColor Gray
}
