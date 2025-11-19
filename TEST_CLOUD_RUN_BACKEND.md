# Guía: Probar Backend en Google Cloud Run

## 🔴 Problema: Request Timeout

Si estás obteniendo "requested timeout" al probar el backend, sigue estos pasos para diagnosticar y solucionar.

## 1. Verificar que el Servicio Esté Corriendo

### Opción A: Google Cloud Console (Más Fácil)

1. **Ve a Cloud Run:**
   - https://console.cloud.google.com/run?project=caria-backend

2. **Busca el servicio `caria-api`**

3. **Verifica el estado:**
   - ✅ **Green/Active** = Servicio corriendo
   - ❌ **Red/Error** = Servicio con problemas
   - ⚠️ **Yellow/Warning** = Servicio iniciando o con warnings

4. **Haz clic en el servicio para ver detalles:**
   - **URL:** Debería mostrar: `https://caria-api-418525923468.us-central1.run.app`
   - **Status:** Debe ser "Active"
   - **Last revision:** Debe tener una revisión reciente

### Opción B: gcloud CLI

```bash
# Ver estado del servicio
gcloud run services describe caria-api \
  --region=us-central1 \
  --project=caria-backend \
  --format="table(status.conditions[0].type,status.conditions[0].status,status.url)"
```

**Salida esperada:**
```
TYPE     STATUS  URL
Ready    True    https://caria-api-418525923468.us-central1.run.app
```

## 2. Probar Health Check (Endpoint Más Simple)

### Desde el Navegador

Abre esta URL directamente:
```
https://caria-api-418525923468.us-central1.run.app/health
```

**Respuesta esperada:**
- ✅ `{"status":"ok"}` o similar = Backend funciona
- ❌ Timeout = Problema de conexión o servicio caído
- ❌ 404 = Endpoint no existe
- ❌ 500 = Error interno

### Desde Terminal (curl)

```bash
# Health check simple
curl https://caria-api-418525923468.us-central1.run.app/health

# Con timeout de 10 segundos
curl --max-time 10 https://caria-api-418525923468.us-central1.run.app/health

# Con más información (verbose)
curl -v https://caria-api-418525923468.us-central1.run.app/health
```

### Desde Python

```python
import requests

try:
    response = requests.get(
        "https://caria-api-418525923468.us-central1.run.app/health",
        timeout=10
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except requests.exceptions.Timeout:
    print("❌ Timeout: El servicio no responde en 10 segundos")
except requests.exceptions.ConnectionError:
    print("❌ Connection Error: No se puede conectar al servicio")
except Exception as e:
    print(f"❌ Error: {e}")
```

## 3. Verificar Logs del Servicio

Los logs te dirán exactamente qué está pasando.

### Opción A: Google Cloud Console

1. **Ve a Cloud Run** → Selecciona `caria-api`
2. **Haz clic en "LOGS"** (pestaña superior)
3. **Busca errores recientes:**
   - Busca líneas en rojo
   - Busca palabras: "error", "timeout", "failed", "exception"

### Opción B: gcloud CLI

```bash
# Ver últimos 50 logs
gcloud run services logs read caria-api \
  --region=us-central1 \
  --project=caria-backend \
  --limit=50

# Ver logs en tiempo real (streaming)
gcloud run services logs tail caria-api \
  --region=us-central1 \
  --project=caria-backend

# Filtrar solo errores
gcloud run services logs read caria-api \
  --region=us-central1 \
  --project=caria-backend \
  --limit=100 | grep -i "error\|exception\|timeout\|failed"
```

### Errores Comunes en Logs

**1. "Container failed to start"**
- Problema: El contenedor no puede iniciar
- Solución: Revisa el Dockerfile y dependencias

**2. "Port 8080 not listening"**
- Problema: La app no está escuchando en el puerto correcto
- Solución: Verifica que la app use `PORT` environment variable

**3. "Database connection failed"**
- Problema: No puede conectar a Cloud SQL
- Solución: Verifica `DATABASE_URL` y Cloud SQL instance

**4. "Secret not found"**
- Problema: Un secret no existe en Secret Manager
- Solución: Crea el secret faltante

**5. "Module not found"**
- Problema: Falta una dependencia Python
- Solución: Verifica `requirements.txt` y Dockerfile

## 4. Verificar Configuración del Servicio

### Revisar Variables de Entorno y Secrets

1. **Cloud Run Console:**
   - Ve a `caria-api` → Última revisión → "VARIABLES & SECRETS"
   - Verifica que todos los secrets estén listados

2. **gcloud CLI:**
```bash
# Ver configuración completa
gcloud run services describe caria-api \
  --region=us-central1 \
  --project=caria-backend \
  --format="yaml"
```

### Verificar Timeout Settings

El timeout por defecto es 300 segundos (5 minutos). Si tu request tarda más, fallará.

```bash
# Ver timeout actual
gcloud run services describe caria-api \
  --region=us-central1 \
  --project=caria-backend \
  --format="value(spec.template.spec.timeoutSeconds)"
```

**Si necesitas aumentar el timeout:**
```bash
gcloud run services update caria-api \
  --region=us-central1 \
  --project=caria-backend \
  --timeout=600
```

## 5. Probar Endpoints Específicos

### Health Check (Sin autenticación)
```bash
curl https://caria-api-418525923468.us-central1.run.app/health
```

### CORS Test (Sin autenticación)
```bash
curl https://caria-api-418525923468.us-central1.run.app/api/cors-test
```

### Endpoints que Requieren Auth (401 esperado)
```bash
# Debe dar 401 (Not authenticated) - esto es normal
curl https://caria-api-418525923468.us-central1.run.app/api/holdings

# Debe dar 401 - esto es normal
curl https://caria-api-418525923468.us-central1.run.app/api/market/fear-greed
```

**Si obtienes 401, el backend funciona pero necesitas autenticación.**

## 6. Diagnosticar Timeout Específico

### Si el Health Check da Timeout:

**Posibles causas:**

1. **Servicio no está corriendo:**
   ```bash
   # Verificar estado
   gcloud run services describe caria-api --region=us-central1 --project=caria-backend
   ```

2. **URL incorrecta:**
   - Verifica que la URL sea: `https://caria-api-418525923468.us-central1.run.app`
   - No uses `http://` (debe ser `https://`)

3. **Firewall/Red bloqueando:**
   - Prueba desde otra red
   - Prueba desde Cloud Shell: https://shell.cloud.google.com

4. **Servicio iniciando (cold start):**
   - Cloud Run puede tardar 10-30 segundos en iniciar si está "frío"
   - Espera 30 segundos y vuelve a intentar

### Si un Endpoint Específico da Timeout:

1. **Revisa los logs** para ese endpoint específico
2. **Verifica que el endpoint exista:**
   ```bash
   # Ver todos los endpoints disponibles
   curl https://caria-api-418525923468.us-central1.run.app/docs
   ```
3. **Verifica que no esté haciendo requests externos lentos:**
   - FMP API puede ser lenta
   - Gemini API puede tardar
   - Reddit API puede tener rate limits

## 7. Usar Cloud Shell para Probar

Si tienes problemas de red local, usa Cloud Shell:

1. **Abre Cloud Shell:**
   - https://shell.cloud.google.com
   - O desde Cloud Console: botón ">_" en la barra superior

2. **Prueba desde ahí:**
   ```bash
   curl https://caria-api-418525923468.us-central1.run.app/health
   ```

3. **Si funciona desde Cloud Shell pero no desde tu máquina:**
   - Problema de red/firewall local
   - Usa VPN o prueba desde otra red

## 8. Script de Diagnóstico Completo

Ejecuta el script que ya creamos:

```bash
cd notebooks
python diagnose_api_connection.py
```

Este script prueba:
- Health check
- CORS
- FMP API
- Gemini API
- Reddit API
- Fear & Greed Index

## 9. Soluciones Rápidas

### Si el servicio no está corriendo:

```bash
# Ver todas las revisiones
gcloud run revisions list \
  --service=caria-api \
  --region=us-central1 \
  --project=caria-backend

# Ver detalles de la última revisión
gcloud run revisions describe caria-api-XXXXX \
  --region=us-central1 \
  --project=caria-backend
```

### Si necesitas redeployar:

```bash
# Trigger nuevo deployment desde GitHub
git commit --allow-empty -m "Trigger: Redeploy to fix timeout"
git push origin main

# O redeploy manual desde Cloud Run Console
# Cloud Run → caria-api → "EDIT & DEPLOY NEW REVISION" → "DEPLOY"
```

### Si el contenedor no inicia:

1. Revisa los logs (paso 3)
2. Verifica que el Dockerfile esté correcto
3. Verifica que todas las dependencias estén en `requirements.txt`

## 10. Checklist de Diagnóstico

- [ ] Servicio aparece como "Active" en Cloud Run Console
- [ ] URL del servicio es correcta
- [ ] Health check responde (no timeout)
- [ ] Logs no muestran errores críticos
- [ ] Variables de entorno y secrets están configurados
- [ ] Timeout del servicio es suficiente (300s o más)
- [ ] Prueba desde Cloud Shell funciona
- [ ] `diagnose_api_connection.py` muestra resultados

## Enlaces Rápidos

- **Cloud Run Console:** https://console.cloud.google.com/run?project=caria-backend
- **Logs:** https://console.cloud.google.com/run/detail/us-central1/caria-api/logs?project=caria-backend
- **Cloud Shell:** https://shell.cloud.google.com
- **Backend URL:** https://caria-api-418525923468.us-central1.run.app/health

## Comandos de Resumen

```bash
# 1. Verificar estado
gcloud run services describe caria-api --region=us-central1 --project=caria-backend

# 2. Ver logs
gcloud run services logs read caria-api --region=us-central1 --project=caria-backend --limit=50

# 3. Probar health check
curl https://caria-api-418525923468.us-central1.run.app/health

# 4. Ver configuración
gcloud run services describe caria-api --region=us-central1 --project=caria-backend --format="yaml"
```

---

**Si después de seguir estos pasos aún tienes timeout, comparte:**
1. El output de `gcloud run services describe caria-api`
2. Los últimos 20 logs del servicio
3. El resultado de `curl https://caria-api-418525923468.us-central1.run.app/health`

