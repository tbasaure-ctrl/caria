# Guía para Configurar API Keys en Cloud Run

## 🔴 Problema Detectado

El diagnóstico muestra que:
- ✅ Backend está corriendo
- ✅ CORS funciona
- ❌ **Reddit API falla con error 401** (credenciales inválidas o no configuradas)
- ⚠️ FMP y Gemini requieren autenticación (normal, pero necesitan secrets configurados)

## 📋 Secrets Requeridos

Los siguientes secrets deben estar en **Google Cloud Secret Manager**:

1. `gemini-api-key` - API Key de Google Gemini
2. `fmp-api-key` - API Key de Financial Modeling Prep
3. `reddit-client-id` - Reddit Client ID: `1eIYr0z6slzt62EXy1KQ6Q`
4. `reddit-client-secret` - Reddit Client Secret: `p53Yud4snfuadHAvgva_6vWkj0eXcw`
5. `postgres-password` - Password de PostgreSQL
6. `jwt-secret-key` - Secret para JWT tokens

## 🔧 Cómo Verificar y Crear Secrets

### Opción 1: Usando Google Cloud Console (Recomendado)

1. **Ve a Secret Manager:**
   - https://console.cloud.google.com/security/secret-manager?project=caria-backend

2. **Verifica si los secrets existen:**
   - Busca cada secret en la lista
   - Si no existe, haz clic en "CREATE SECRET"

3. **Crear un secret (ejemplo Reddit Client ID):**
   - Name: `reddit-client-id`
   - Secret value: `1eIYr0z6slzt62EXy1KQ6Q`
   - Haz clic en "CREATE SECRET"

4. **Repite para todos los secrets faltantes**

### Opción 2: Usando gcloud CLI

Si tienes `gcloud` instalado y autenticado:

```bash
# Autenticarse
gcloud auth login
gcloud config set project caria-backend

# Crear secrets
echo -n 'TU_GEMINI_API_KEY' | gcloud secrets create gemini-api-key --data-file=- --project=caria-backend
echo -n 'TU_FMP_API_KEY' | gcloud secrets create fmp-api-key --data-file=- --project=caria-backend
echo -n '1eIYr0z6slzt62EXy1KQ6Q' | gcloud secrets create reddit-client-id --data-file=- --project=caria-backend
echo -n 'p53Yud4snfuadHAvgva_6vWkj0eXcw' | gcloud secrets create reddit-client-secret --data-file=- --project=caria-backend
echo -n 'TU_POSTGRES_PASSWORD' | gcloud secrets create postgres-password --data-file=- --project=caria-backend
echo -n 'TU_JWT_SECRET' | gcloud secrets create jwt-secret-key --data-file=- --project=caria-backend
```

### Opción 3: Verificar Secrets Existentes

```bash
# Listar todos los secrets
gcloud secrets list --project=caria-backend

# Ver detalles de un secret específico
gcloud secrets describe reddit-client-id --project=caria-backend

# Ver versiones de un secret
gcloud secrets versions list reddit-client-id --project=caria-backend
```

## ✅ Verificar que el Workflow los Está Pasando

El workflow en `.github/workflows/deploy-cloud-run.yml` debe incluir:

```yaml
--set-secrets=GEMINI_API_KEY=gemini-api-key:latest,POSTGRES_PASSWORD=postgres-password:latest,JWT_SECRET_KEY=jwt-secret-key:latest,FMP_API_KEY=fmp-api-key:latest,REDDIT_CLIENT_ID=reddit-client-id:latest,REDDIT_CLIENT_SECRET=reddit-client-secret:latest
```

**✅ Esto ya está configurado correctamente en el workflow actual.**

## 🔍 Verificar en Cloud Run

Después de crear los secrets, verifica que Cloud Run los esté usando:

1. **Ve a Cloud Run:**
   - https://console.cloud.google.com/run?project=caria-backend

2. **Selecciona el servicio `caria-api`**

3. **Ve a la pestaña "REVISIONS AND TRAFFIC"**

4. **Haz clic en la revisión más reciente**

5. **Ve a "VARIABLES & SECRETS"**

6. **Verifica que todos los secrets estén listados:**
   - `GEMINI_API_KEY` → `gemini-api-key:latest`
   - `FMP_API_KEY` → `fmp-api-key:latest`
   - `REDDIT_CLIENT_ID` → `reddit-client-id:latest`
   - `REDDIT_CLIENT_SECRET` → `reddit-client-secret:latest`
   - `POSTGRES_PASSWORD` → `postgres-password:latest`
   - `JWT_SECRET_KEY` → `jwt-secret-key:latest`

## 🚀 Después de Crear los Secrets

1. **Haz un nuevo deployment:**
   - Puedes hacer un commit vacío para trigger el workflow:
     ```bash
     git commit --allow-empty -m "Trigger: Redeploy after adding secrets"
     git push origin main
     ```

2. **O redeploy manualmente desde Cloud Run:**
   - Ve a Cloud Run → caria-api → "EDIT & DEPLOY NEW REVISION"
   - No cambies nada, solo haz clic en "DEPLOY"
   - Esto aplicará los secrets existentes

## 🧪 Probar Después del Deployment

Ejecuta el script de diagnóstico:

```bash
python diagnose_api_connection.py
```

Deberías ver:
- ✅ Reddit API funciona correctamente
- ✅ FMP API funciona (después de login)
- ✅ Gemini API funciona (después de login)

## 📝 Valores de Reddit (Ya Conocidos)

- **Client ID:** `1eIYr0z6slzt62EXy1KQ6Q`
- **Client Secret:** `p53Yud4snfuadHAvgva_6vWkj0eXcw`
- **User Agent:** `Caria-Investment-App-v1.0` (ya configurado como env var)

## ⚠️ Notas Importantes

1. **Los secrets deben tener la versión `:latest`** en el workflow para que siempre use la versión más reciente.

2. **Si cambias un secret**, necesitas crear una nueva versión:
   ```bash
   echo -n 'NUEVO_VALOR' | gcloud secrets versions add SECRET_NAME --data-file=- --project=caria-backend
   ```

3. **El servicio de Cloud Run necesita permisos** para acceder a Secret Manager. Esto debería estar configurado automáticamente, pero si hay problemas, verifica:
   - Service Account: `caria-api@caria-backend.iam.gserviceaccount.com`
   - Role: `roles/secretmanager.secretAccessor`

## 🔗 Enlaces Útiles

- **Secret Manager:** https://console.cloud.google.com/security/secret-manager?project=caria-backend
- **Cloud Run:** https://console.cloud.google.com/run?project=caria-backend
- **Logs de Cloud Run:** https://console.cloud.google.com/logs/query?project=caria-backend

## 📞 Si Aún No Funciona

1. **Revisa los logs de Cloud Run:**
   ```bash
   gcloud run services logs read caria-api --region=us-central1 --limit=50
   ```

2. **Busca errores relacionados con:**
   - "Secret not found"
   - "Permission denied"
   - "401 Unauthorized" (Reddit)
   - "API key invalid"

3. **Verifica que el service account tenga permisos:**
   ```bash
   gcloud projects get-iam-policy caria-backend --flatten="bindings[].members" --filter="bindings.members:serviceAccount:caria-api@caria-backend.iam.gserviceaccount.com"
   ```

