# ⚡ Solución Rápida al Error de Deploy

## 🎯 El Problema

Estás en el directorio correcto (`services/functions`), pero las APIs de Google Cloud aún no están habilitadas completamente.

## ✅ Solución Paso a Paso

### Paso 1: Habilitar APIs Manualmente (RECOMENDADO)

1. **Ve a Google Cloud Console:**
   - Abre: https://console.cloud.google.com/apis/library?project=caria-9b633
   - O ve a: https://console.cloud.google.com → Selecciona proyecto "caria-9b633" → APIs & Services → Library

2. **Habilita estas APIs (una por una):**
   - Busca **"Cloud Functions API"** → Click **"Enable"**
   - Busca **"Cloud Build API"** → Click **"Enable"**
   - Busca **"Artifact Registry API"** → Click **"Enable"**
   - Busca **"Cloud Logging API"** → Click **"Enable"**

3. **Espera 2-3 minutos** después de habilitar todas

4. **Vuelve a intentar:**
   ```powershell
   cd C:\key\wise_adviser_cursor_context\notebooks\services\functions
   firebase deploy --only functions
   ```

### Paso 2: Verificar Facturación

1. Ve a: https://console.firebase.google.com/project/caria-9b633/settings/usage
2. Verifica que veas **"Blaze plan"** activo
3. Si dice "Spark plan", haz click en **"Upgrade"**

### Paso 3: Si Aún Falla - Usar gcloud CLI

Si tienes Google Cloud SDK instalado:

```powershell
# Habilitar APIs desde la línea de comandos
gcloud services enable cloudfunctions.googleapis.com --project=caria-9b633
gcloud services enable cloudbuild.googleapis.com --project=caria-9b633
gcloud services enable artifactregistry.googleapis.com --project=caria-9b633
gcloud services enable logging.googleapis.com --project=caria-9b633

# Esperar 2 minutos y reintentar
firebase deploy --only functions
```

### Paso 4: Verificar con Debug

Para ver más detalles del error:

```powershell
firebase deploy --only functions --debug
```

Esto te mostrará exactamente qué está fallando.

---

## 📋 Comandos Correctos

**IMPORTANTE:** Siempre ejecuta estos comandos desde `services/functions`:

```powershell
# Navegar al directorio correcto
cd C:\key\wise_adviser_cursor_context\notebooks\services\functions

# Verificar que estás en el lugar correcto
ls firebase.json  # Debe mostrar el archivo

# Hacer deploy
firebase deploy --only functions
```

---

## 🔍 Verificar Estado de las APIs

Puedes verificar qué APIs están habilitadas:

1. Ve a: https://console.cloud.google.com/apis/dashboard?project=caria-9b633
2. Busca las APIs mencionadas arriba
3. Verifica que digan **"Enabled"**

---

## ⏰ Tiempo de Espera

Después de habilitar las APIs, pueden tardar:
- **Mínimo:** 1-2 minutos
- **Normal:** 3-5 minutos
- **Máximo:** 10-15 minutos

**Sé paciente** y espera antes de reintentar.

---

## 🆘 Si Nada Funciona

1. **Espera 15 minutos** y vuelve a intentar
2. **Cierra y abre** Firebase CLI:
   ```powershell
   firebase logout
   firebase login
   ```
3. **Contacta soporte:** https://firebase.google.com/support

---

## ✅ Checklist

- [ ] Estoy en `services/functions` (no en `services`)
- [ ] Habilité las 4 APIs en Google Cloud Console
- [ ] Esperé 2-3 minutos después de habilitar
- [ ] Verifiqué que el plan Blaze esté activo
- [ ] Reintenté el deploy

---

## 💡 Nota Importante

El error **"An unexpected error has occurred"** casi siempre se resuelve:
1. Habilitando las APIs manualmente
2. Esperando unos minutos
3. Reintentando

**No es un error de tu código**, es solo que las APIs necesitan tiempo para habilitarse.

