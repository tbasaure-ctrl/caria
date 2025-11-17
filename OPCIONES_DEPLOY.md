# OPCIONES PARA DESPLEGAR SIN PROBLEMAS

## Opción 1: Dockerfile Simplificado (RECOMENDADO)
He creado `services/Dockerfile.SIMPLE` que copia TODO sin exclusiones.

**Pasos:**
1. Renombra el Dockerfile actual:
   ```cmd
   cd C:\key\wise_adviser_cursor_context\notebooks
   ren services\Dockerfile Dockerfile.old
   ren services\Dockerfile.SIMPLE Dockerfile
   ```

2. Reconstruye y despliega:
   ```cmd
   gcloud builds submit --tag us-central1-docker.pkg.dev/caria-backend/caria-api-repo/caria-api:latest
   DEPLOY_AHORA.bat
   ```

---

## Opción 2: Usar Google Cloud Shell (Más fácil)

**Pasos:**
1. Ve a: https://shell.cloud.google.com/
2. Sube tus archivos:
   ```bash
   # En Cloud Shell, ejecuta:
   gcloud config set project caria-backend
   
   # Crea un directorio temporal
   mkdir -p ~/caria-deploy
   cd ~/caria-deploy
   ```

3. Sube los archivos desde tu máquina:
   - En Cloud Shell, haz clic en el ícono de "Subir archivo" (icono de carpeta con flecha arriba)
   - Sube la carpeta completa `notebooks/`
   - O usa `gcloud cloud-shell scp` si tienes muchos archivos

4. Construye desde Cloud Shell:
   ```bash
   cd notebooks
   gcloud builds submit --tag us-central1-docker.pkg.dev/caria-backend/caria-api-repo/caria-api:latest
   ```

---

## Opción 3: Conectar Repositorio Git (Más profesional)

**Pasos:**
1. Sube tu código a GitHub/GitLab
2. Ve a Cloud Build en Google Cloud Console
3. Conecta tu repositorio
4. Cloud Build construirá automáticamente cada vez que hagas push

**Ventajas:**
- Automático
- Historial de builds
- Fácil de mantener

---

## Opción 4: Usar Cloud Build con archivo de configuración

Ya tienes `cloudbuild.yaml`. Puedes ejecutarlo directamente:

```cmd
gcloud builds submit --config=cloudbuild.yaml
```

Esto usa la configuración en `cloudbuild.yaml` que ya tienes.

---

## Opción 5: Construir localmente y subir

Si tienes Docker instalado localmente:

```cmd
# Construir imagen localmente
docker build -t caria-api:latest -f services/Dockerfile .

# Etiquetar para Artifact Registry
docker tag caria-api:latest us-central1-docker.pkg.dev/caria-backend/caria-api-repo/caria-api:latest

# Autenticar Docker
gcloud auth configure-docker us-central1-docker.pkg.dev

# Subir imagen
docker push us-central1-docker.pkg.dev/caria-backend/caria-api-repo/caria-api:latest

# Desplegar
DEPLOY_AHORA.bat
```

---

## RECOMENDACIÓN

**Empieza con Opción 1** (Dockerfile simplificado). Es la más rápida y debería funcionar.

Si no funciona, prueba **Opción 2** (Cloud Shell) - es más fácil porque no tienes que lidiar con `.dockerignore` ni paths locales.

