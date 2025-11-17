# Script de setup para Firebase Functions (PowerShell)
# Para Windows

Write-Host "🚀 Configurando Firebase Functions para Wise Adviser" -ForegroundColor Cyan
Write-Host ""

# Verificar si Firebase CLI está instalado
$firebaseCmd = Get-Command firebase -ErrorAction SilentlyContinue
if (-not $firebaseCmd) {
    Write-Host "❌ Firebase CLI no está instalado." -ForegroundColor Red
    Write-Host "Instala con: npm install -g firebase-tools" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Firebase CLI encontrado" -ForegroundColor Green
Write-Host ""

# Login a Firebase
Write-Host "📝 Iniciando sesión en Firebase..." -ForegroundColor Cyan
firebase login

Write-Host ""
Write-Host "🔧 Configurando variables de entorno..." -ForegroundColor Cyan
Write-Host ""

$GEMINI_KEY = Read-Host "GEMINI_API_KEY"
$BACKEND_URL = Read-Host "BACKEND_URL (default: http://localhost:8000)"
if ([string]::IsNullOrWhiteSpace($BACKEND_URL)) {
    $BACKEND_URL = "http://localhost:8000"
}

$SETUP_LLAMA = Read-Host "¿Quieres configurar Llama como fallback? (y/n)"

# Configurar Gemini
firebase functions:config:set gemini.api_key="$GEMINI_KEY"
firebase functions:config:set gemini.api_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
firebase functions:config:set backend.url="$BACKEND_URL"

# Configurar Llama si el usuario quiere
if ($SETUP_LLAMA -eq "y" -or $SETUP_LLAMA -eq "Y") {
    $LLAMA_KEY = Read-Host "LLAMA_API_KEY"
    $LLAMA_URL = Read-Host "LLAMA_API_URL"
    $LLAMA_MODEL = Read-Host "LLAMA_MODEL_NAME (default: llama-3.1-70b-instruct)"
    if ([string]::IsNullOrWhiteSpace($LLAMA_MODEL)) {
        $LLAMA_MODEL = "llama-3.1-70b-instruct"
    }
    
    firebase functions:config:set llama.api_key="$LLAMA_KEY"
    firebase functions:config:set llama.api_url="$LLAMA_URL"
    firebase functions:config:set llama.model_name="$LLAMA_MODEL"
}

Write-Host ""
Write-Host "✅ Configuración completada!" -ForegroundColor Green
Write-Host ""
Write-Host "Para desplegar las funciones, ejecuta:" -ForegroundColor Cyan
Write-Host "  firebase deploy --only functions" -ForegroundColor Yellow
Write-Host ""

