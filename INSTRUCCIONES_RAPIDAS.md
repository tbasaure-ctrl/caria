# 🚀 Instrucciones Rápidas - GitHub Actions

## Paso 1: Ejecutar Script de Configuración en Cloud Shell

1. Abre Cloud Shell: https://shell.cloud.google.com/
2. Clona el repositorio:
   ```bash
   cd ~
   git clone https://github.com/tbasaure-ctrl/caria.git
   cd caria
   ```
3. Ejecuta el script de configuración:
   ```bash
   
   ```

Este script configura automáticamente:
- ✅ Service Account
- ✅ Permisos necesarios
- ✅ Workload Identity Pool
- ✅ Provider de GitHub
- ✅ Vinculación entre Service Account y GitHub

## Paso 2: Subir el Workflow a GitHub

Desde tu máquina local:

```bash
cd C:\key\wise_adviser_cursor_context\notebooks

# Asegúrate de estar en la rama main
git checkout main

# Agregar el workflow
git add .github/workflows/deploy-cloud-run.yml

# Commit
git commit -m "Add GitHub Actions workflow for Cloud Run"

# Push
git push origin main
```

## Paso 3: Verificar Despliegue

Una vez que hagas push, GitHub Actions automáticamente:
1. Construirá la imagen Docker
2. La subirá a Artifact Registry
3. Desplegará a Cloud Run

Puedes ver el progreso en:
**https://github.com/tbasaure-ctrl/caria/actions**

## ✅ Listo!

Después del primer despliegue exitoso, cada vez que hagas `git push origin main`, se desplegará automáticamente.





