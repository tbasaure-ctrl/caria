# Script helper de PowerShell para ejecutar scripts de Caria desde cualquier lugar
# Uso: .\run_script.ps1 scripts/orchestration/run_regime_hmm.py [argumentos]

param(
    [Parameter(Mandatory=$true)]
    [string]$ScriptPath,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ScriptArgs
)

# Obtener directorio del script actual (caria_data/)
$CariaDataDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Cambiar al directorio caria_data
Set-Location $CariaDataDir

# Ejecutar el script con los argumentos proporcionados
Write-Host "Ejecutando desde: $CariaDataDir" -ForegroundColor Green
Write-Host "Script: $ScriptPath" -ForegroundColor Green
Write-Host "Argumentos: $($ScriptArgs -join ' ')" -ForegroundColor Green
Write-Host ""

python $ScriptPath $ScriptArgs

