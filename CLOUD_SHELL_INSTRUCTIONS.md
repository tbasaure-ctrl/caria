# 🚀 Instrucciones para Desplegar desde Google Cloud Shell

## Paso 1: Abrir Cloud Shell

1. Ve a: https://console.cloud.google.com/
2. Click en el ícono de terminal (☁️) en la esquina superior derecha
3. O ve directamente a: https://shell.cloud.google.com/

## Paso 2: Subir Archivos

Tienes 3 opciones:

### Opción A: Subir desde la interfaz (Más fácil)
1. En Cloud Shell, click en el ícono de carpeta con flecha arriba (☁️↑)
2. Selecciona la carpeta `notebooks` completa
3. Espera a que termine la subida

### Opción B: Usar gcloud cloud-shell scp
```bash
# Desde tu máquina local (PowerShell)
gcloud cloud-shell scp --recurse C:\key\wise_adviser_cursor_context\notebooks cloud-shell:~/notebooks
```

### Opción C: Clonar desde Git (Si ya subiste a GitHub)
```bash
cd ~
git clone https://github.com/tbasaure-ctrl/caria.git
cd caria
```

## Paso 3: Ejecutar Script de Despliegue

Una vez que tengas los archivos en Cloud Shell:

```bash
# Copia y pega este comando completo:
cd ~/notebooks  # o donde hayas subido los archivos

# Copia el contenido de DEPLOY_CLOUD_SHELL.sh y ejecútalo:
bash <(cat << 'EOF'
# [Pega aquí todo el contenido del script]
EOF
)
```

**O mejor aún**, crea el archivo y ejecútalo:

```bash
# Crear el script
nano deploy.sh
# [Pega el contenido de DEPLOY_CLOUD_SHELL.sh]
# Guarda con Ctrl+O, Enter, Ctrl+X

# Hacer ejecutable
chmod +x deploy.sh

# Ejecutar
./deploy.sh
```

## Paso 4: Verificar Despliegue

```bash
# Ver URL del servicio
gcloud run services describe caria-api --region=us-central1 --format="value(status.url)"

# Ver logs
gcloud run services logs read caria-api --region=us-central1 --limit=50

# Probar endpoint
curl $(gcloud run services describe caria-api --region=us-central1 --format="value(status.url)")/health
```

## ⚠️ Si algo falla

1. **Build falla**: Revisa los logs del build en Cloud Console
2. **Deploy falla**: Revisa los logs del servicio con el comando de arriba
3. **Módulo no encontrado**: Verifica que el Dockerfile copie correctamente `caria_data/src/`

## 📝 Notas Importantes

- El script usa el Dockerfile que está en `services/Dockerfile`
- Asegúrate de que el `.dockerignore` no esté excluyendo `caria_data/src/caria/models/`
- El build puede tardar 15-20 minutos la primera vez

