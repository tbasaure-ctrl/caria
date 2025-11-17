# Script para diagnosticar y resolver problemas de deploy de Firebase Functions

Write-Host "🔍 Diagnosticando problema de deploy..." -ForegroundColor Cyan
Write-Host ""

# 1. Verificar login
Write-Host "1. Verificando login de Firebase..." -ForegroundColor Yellow
firebase login:list
Write-Host ""

# 2. Verificar proyecto
Write-Host "2. Verificando proyecto actual..." -ForegroundColor Yellow
firebase use
Write-Host ""

# 3. Verificar configuración
Write-Host "3. Verificando configuración de funciones..." -ForegroundColor Yellow
if (Test-Path "firebase.json") {
    Write-Host "✅ firebase.json encontrado" -ForegroundColor Green
} else {
    Write-Host "❌ firebase.json NO encontrado" -ForegroundColor Red
}

if (Test-Path "functions\main.py") {
    Write-Host "✅ functions/main.py encontrado" -ForegroundColor Green
} else {
    Write-Host "❌ functions/main.py NO encontrado" -ForegroundColor Red
}

if (Test-Path "functions\requirements.txt") {
    Write-Host "✅ functions/requirements.txt encontrado" -ForegroundColor Green
} else {
    Write-Host "❌ functions/requirements.txt NO encontrado" -ForegroundColor Red
}
Write-Host ""

# 4. Verificar variables de entorno
Write-Host "4. Verificando variables de entorno..." -ForegroundColor Yellow
firebase functions:config:get
Write-Host ""

# 5. Instrucciones
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "SOLUCIONES RECOMENDADAS:" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "OPCIÓN 1: Esperar y Reintentar (Más común)" -ForegroundColor Yellow
Write-Host "  Las APIs pueden tardar 2-5 minutos en habilitarse."
Write-Host "  Espera unos minutos y ejecuta:" -ForegroundColor White
Write-Host "    firebase deploy --only functions" -ForegroundColor Green
Write-Host ""
Write-Host "OPCIÓN 2: Habilitar APIs Manualmente" -ForegroundColor Yellow
Write-Host "  1. Ve a: https://console.cloud.google.com/apis/library"
Write-Host "  2. Selecciona proyecto: caria-9b633"
Write-Host "  3. Busca y habilita:" -ForegroundColor White
Write-Host "     - Cloud Functions API" -ForegroundColor Green
Write-Host "     - Cloud Build API" -ForegroundColor Green
Write-Host "     - Artifact Registry API" -ForegroundColor Green
Write-Host "  4. Espera 1-2 minutos y vuelve a intentar"
Write-Host ""
Write-Host "OPCIÓN 3: Verificar Facturación" -ForegroundColor Yellow
Write-Host "  1. Ve a: https://console.firebase.google.com/project/caria-9b633/settings/usage"
Write-Host "  2. Verifica que el plan Blaze esté activo"
Write-Host "  3. Si no está activo, haz click en 'Upgrade'"
Write-Host ""
Write-Host "OPCIÓN 4: Deploy con Debug" -ForegroundColor Yellow
Write-Host "  Para ver más detalles del error:" -ForegroundColor White
Write-Host "    firebase deploy --only functions --debug" -ForegroundColor Green
Write-Host ""

