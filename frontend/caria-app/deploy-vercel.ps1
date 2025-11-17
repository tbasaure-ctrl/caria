# Script de deploy a Vercel para Windows PowerShell

Write-Host "🚀 Deploy a Vercel - Wise Adviser" -ForegroundColor Cyan
Write-Host ""

# Verificar si Vercel CLI está instalado
$vercelCmd = Get-Command vercel -ErrorAction SilentlyContinue
if (-not $vercelCmd) {
    Write-Host "📦 Instalando Vercel CLI..." -ForegroundColor Yellow
    npm install -g vercel
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error instalando Vercel CLI" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ Vercel CLI encontrado" -ForegroundColor Green
Write-Host ""

# Verificar si está logueado
Write-Host "🔐 Verificando login..." -ForegroundColor Yellow
vercel whoami 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "📝 Necesitas hacer login..." -ForegroundColor Yellow
    vercel login
}

Write-Host ""
Write-Host "📋 Configuración:" -ForegroundColor Cyan
Write-Host "  - Root Directory: caria_data/caria-app" -ForegroundColor White
Write-Host "  - Framework: Vite" -ForegroundColor White
Write-Host "  - Build Command: npm run build" -ForegroundColor White
Write-Host ""

# Preguntar por variables de entorno
Write-Host "🔧 Variables de Entorno:" -ForegroundColor Cyan
$apiUrl = Read-Host "VITE_API_URL (default: http://localhost:8000)"
if ([string]::IsNullOrWhiteSpace($apiUrl)) {
    $apiUrl = "http://localhost:8000"
}

Write-Host ""
Write-Host "🚀 Iniciando deploy..." -ForegroundColor Cyan
Write-Host ""

# Deploy
vercel --prod

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Deploy completado!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📝 No olvides configurar las variables de entorno en Vercel Dashboard:" -ForegroundColor Yellow
    Write-Host "   - VITE_API_URL = $apiUrl" -ForegroundColor White
    Write-Host ""
    Write-Host "   Ve a: https://vercel.com/dashboard → Tu Proyecto → Settings → Environment Variables" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "❌ Error en el deploy. Revisa los logs arriba." -ForegroundColor Red
}

