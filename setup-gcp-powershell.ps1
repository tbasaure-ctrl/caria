# Script de setup para Google Cloud Platform (PowerShell)
# Uso: .\setup-gcp-powershell.ps1

Write-Host "🚀 Setup de Google Cloud Platform para Caria" -ForegroundColor Blue
Write-Host ""

# Verificar que gcloud está disponible
try {
    $gcloudVersion = gcloud --version 2>&1 | Select-Object -First 1
    Write-Host "✓ gcloud encontrado: $gcloudVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ gcloud no está en el PATH" -ForegroundColor Red
    Write-Host "   Agrega gcloud al PATH o ejecuta desde: C:\Users\tomas\AppData\Local\Google\Cloud SDK" -ForegroundColor Yellow
    Write-Host "   O instala desde: https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

# Configurar proyecto
$PROJECT_ID = Read-Host "ID del proyecto GCP (default: 418525923468)"
if ([string]::IsNullOrWhiteSpace($PROJECT_ID)) {
    $PROJECT_ID = "418525923468"
}
gcloud config set project $PROJECT_ID
Write-Host "✓ Proyecto configurado: $PROJECT_ID" -ForegroundColor Green
Write-Host ""

# Verificar facturación
Write-Host "Verificando facturación..." -ForegroundColor Blue
$billingEnabled = gcloud billing projects describe $PROJECT_ID --format="value(billingEnabled)" 2>$null
if ($billingEnabled -ne "true") {
    Write-Host "❌ Facturación no habilitada" -ForegroundColor Red
    Write-Host "   Ve a: https://console.cloud.google.com/billing" -ForegroundColor Yellow
    Write-Host "   O ejecuta: gcloud billing projects link $PROJECT_ID --billing-account=BILLING_ACCOUNT_ID" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Facturación habilitada" -ForegroundColor Green
Write-Host ""

# Crear Cloud SQL
Write-Host "Configurando Cloud SQL..." -ForegroundColor Blue
$createDb = Read-Host "¿Crear nueva instancia de Cloud SQL? (y/n)"
if ($createDb -eq "y") {
    $dbInstance = Read-Host "Nombre de la instancia (default: caria-db)"
    if ([string]::IsNullOrWhiteSpace($dbInstance)) {
        $dbInstance = "caria-db"
    }
    
    $region = Read-Host "Región (default: us-central1)"
    if ([string]::IsNullOrWhiteSpace($region)) {
        $region = "us-central1"
    }
    
    $dbPassword = Read-Host "Password para root (mínimo 8 caracteres)" -AsSecureString
    $dbPasswordPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($dbPassword))
    
    Write-Host "Creando instancia (esto puede tomar 5-10 minutos)..." -ForegroundColor Yellow
    gcloud sql instances create $dbInstance `
        --database-version=POSTGRES_15 `
        --tier=db-f1-micro `
        --region=$region `
        --root-password=$dbPasswordPlain `
        --project=$PROJECT_ID
    
    # Crear base de datos
    gcloud sql databases create caria --instance=$dbInstance --project=$PROJECT_ID
    
    # Obtener connection name
    $connectionName = gcloud sql instances describe $dbInstance --format="value(connectionName)" --project=$PROJECT_ID
    Write-Host "✓ Cloud SQL creado: $connectionName" -ForegroundColor Green
} else {
    $dbInstance = Read-Host "Nombre de la instancia existente"
    $connectionName = gcloud sql instances describe $dbInstance --format="value(connectionName)" --project=$PROJECT_ID
    Write-Host "✓ Usando instancia existente: $connectionName" -ForegroundColor Green
}
Write-Host ""

# Configurar Secret Manager
Write-Host "Configurando Secret Manager..." -ForegroundColor Blue
$hasKey = Read-Host "¿Tienes una Gemini API Key? (y/n)"
if ($hasKey -eq "y") {
    $geminiKey = Read-Host "Gemini API Key" -AsSecureString
    $geminiKeyPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($geminiKey))
    
    # Crear secret
    $geminiKeyPlain | gcloud secrets create gemini-api-key --data-file=- --project=$PROJECT_ID 2>$null
    if ($LASTEXITCODE -ne 0) {
        # Secret ya existe, agregar versión
        $geminiKeyPlain | gcloud secrets versions add gemini-api-key --data-file=- --project=$PROJECT_ID
    }
    
    # Dar permisos a Cloud Run
    $projectNumber = gcloud projects describe $PROJECT_ID --format="value(projectNumber)"
    gcloud secrets add-iam-policy-binding gemini-api-key `
        --member="serviceAccount:$projectNumber-compute@developer.gserviceaccount.com" `
        --role="roles/secretmanager.secretAccessor" `
        --project=$PROJECT_ID
    
    Write-Host "✓ Secret creado" -ForegroundColor Green
} else {
    Write-Host "⚠️  Obtén una key en: https://makersuite.google.com/app/apikey" -ForegroundColor Yellow
    Write-Host "   Luego ejecuta:" -ForegroundColor Yellow
    Write-Host "   echo -n 'tu-key' | gcloud secrets create gemini-api-key --data-file=-" -ForegroundColor Yellow
}
Write-Host ""

# Resumen
Write-Host "✅ Configuración completada!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Variables para usar:" -ForegroundColor Blue
Write-Host "   PROJECT_ID=$PROJECT_ID"
Write-Host "   CLOUDSQL_INSTANCE=$connectionName"
Write-Host ""
Write-Host "🚀 Próximo paso:" -ForegroundColor Blue
Write-Host "   1. Construir imagen: gcloud builds submit --tag gcr.io/$PROJECT_ID/caria-api" -ForegroundColor Yellow
Write-Host "   2. Desplegar: Ver GCP_SETUP_PASO_A_PASO.md" -ForegroundColor Yellow
Write-Host ""
Write-Host "   O ejecuta los comandos del Paso 4 en GCP_SETUP_PASO_A_PASO.md" -ForegroundColor Yellow

