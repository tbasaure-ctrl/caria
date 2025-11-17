# Script para inicializar Git y subir a GitHub

Write-Host "🚀 Setup GitHub para Wise Adviser" -ForegroundColor Cyan
Write-Host ""

# Verificar si Git está instalado
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Write-Host "❌ Git no está instalado. Instala Git desde: https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Git encontrado" -ForegroundColor Green
Write-Host ""

# Verificar si ya es un repo Git
if (Test-Path ".git") {
    Write-Host "⚠️  Ya es un repositorio Git" -ForegroundColor Yellow
    $continue = Read-Host "¿Quieres continuar de todas formas? (y/n)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 0
    }
} else {
    Write-Host "📦 Inicializando Git..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Git inicializado" -ForegroundColor Green
}

Write-Host ""
Write-Host "📋 Pasos siguientes:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Crea un repositorio en GitHub:" -ForegroundColor Yellow
Write-Host "   https://github.com/new" -ForegroundColor White
Write-Host ""
Write-Host "2. NO marques Initialize with README" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Copia la URL del repositorio (ej: https://github.com/usuario/repo.git)" -ForegroundColor Yellow
Write-Host ""

$repoUrl = Read-Host "Pega la URL del repositorio de GitHub"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host "❌ URL no proporcionada" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "📝 Agregando archivos..." -ForegroundColor Yellow
git add .

Write-Host "💾 Creando commit inicial..." -ForegroundColor Yellow
git commit -m "Initial commit: Wise Adviser with Firebase Functions and Vercel config"

Write-Host "🔗 Agregando remote..." -ForegroundColor Yellow
git remote add origin $repoUrl 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    # Si ya existe, actualizar
    git remote set-url origin $repoUrl
}

Write-Host "🌿 Configurando branch main..." -ForegroundColor Yellow
git branch -M main

Write-Host ""
Write-Host "🚀 Subiendo a GitHub..." -ForegroundColor Cyan
Write-Host "   (Puede pedirte credenciales de GitHub)" -ForegroundColor Yellow
Write-Host ""

git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ ¡Proyecto subido a GitHub exitosamente!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📝 Próximos pasos:" -ForegroundColor Cyan
    Write-Host "   1. Ve a: https://vercel.com/new" -ForegroundColor White
    Write-Host "   2. Importa tu repositorio de GitHub" -ForegroundColor White
    Write-Host "   3. Configura Root Directory: caria_data/caria-app" -ForegroundColor White
    Write-Host "   4. Agrega variable VITE_API_URL" -ForegroundColor White
    Write-Host "   5. Deploy!" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "❌ Error al subir. Verifica:" -ForegroundColor Red
    Write-Host "   - Que el repositorio exista en GitHub" -ForegroundColor Yellow
    Write-Host "   - Que tengas permisos para escribir" -ForegroundColor Yellow
    Write-Host "   - Que hayas hecho login en Git" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "💡 Si es la primera vez, puede que necesites configurar Git:" -ForegroundColor Cyan
    Write-Host '   git config --global user.name "Tu Nombre"' -ForegroundColor White
    Write-Host '   git config --global user.email "tu@email.com"' -ForegroundColor White
}
