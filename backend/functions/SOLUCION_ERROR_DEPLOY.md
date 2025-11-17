# 🔧 Solución al Error "An unexpected error has occurred"

## 🎯 Solución Rápida (Intenta esto primero)

### Paso 1: Esperar 2-3 minutos

Las APIs de Google Cloud pueden tardar en habilitarse. **Espera 2-3 minutos** y vuelve a intentar:

```powershell
firebase deploy --only functions
```

---

### Paso 2: Habilitar APIs Manualmente (Si Paso 1 no funciona)

1. Ve a [Google Cloud Console - APIs](https://console.cloud.google.com/apis/library?project=caria-9b633)
2. Busca y habilita estas APIs (una por una):
   - **Cloud Functions API** → Click "Enable"
   - **Cloud Build API** → Click "Enable"  
   - **Artifact Registry API** → Click "Enable"
   - **Cloud Logging API** → Click "Enable"
3. Espera 1-2 minutos después de habilitar todas
4. Vuelve a intentar el deploy

---

### Paso 3: Verificar Facturación

1. Ve a [Firebase Console - Billing](https://console.firebase.google.com/project/caria-9b633/settings/usage)
2. Verifica que veas **"Blaze plan"** activo
3. Si dice "Spark plan", haz click en **"Upgrade"** o **"Modify plan"**
4. Confirma la actualización a Blaze (pay-as-you-go)

---

### Paso 4: Configurar Variable Faltante

Veo que falta configurar `backend.url`. Ejecuta:

```powershell
firebase functions:config:set backend.url="http://localhost:8000"
```

---

### Paso 5: Reintentar con Debug

Si aún falla, ejecuta con más información:

```powershell
firebase deploy --only functions --debug
```

Esto te mostrará exactamente dónde está fallando.

---

## 🔍 Diagnóstico Detallado

Ejecuta este script para diagnosticar:

```powershell
.\fix_deploy.ps1
```

---

## ✅ Solución Alternativa: Usar Variables de Entorno Modernas

Firebase está deprecando `functions.config()`. Puedes migrar a variables de entorno modernas:

### Crear archivo `.env` (para desarrollo local)

Crea `functions/.env`:

```env
GEMINI_API_KEY=AIzaSyC-EeIteUCY3gh0z4eFqRiwnqqkO9E5RQU
GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent
BACKEND_URL=http://localhost:8000
```

### Actualizar `main.py` para usar variables de entorno

El código ya está preparado para usar `os.environ.get()`, así que funcionará tanto con `functions.config()` como con variables de entorno.

Para producción, configura las variables en Firebase Console:
1. Firebase Console → Functions → Config
2. Agrega variables de entorno allí

---

## 🆘 Si Nada Funciona

1. **Espera 10-15 minutos** - A veces las APIs tardan en propagarse completamente
2. **Intenta desde Google Cloud Console directamente:**
   - Ve a [Cloud Functions](https://console.cloud.google.com/functions?project=caria-9b633)
   - Click en "Create Function"
   - Esto forzará la habilitación de todas las APIs necesarias
3. **Contacta soporte:** [Firebase Support](https://firebase.google.com/support)

---

## 📋 Checklist

- [ ] Esperé 2-3 minutos después del primer intento
- [ ] Habilité las APIs manualmente en Google Cloud Console
- [ ] Verifiqué que el plan Blaze esté activo
- [ ] Configuré `backend.url` con `firebase functions:config:set`
- [ ] Intenté con `--debug` para ver más detalles
- [ ] Verifiqué permisos en Google Cloud Console → IAM

---

## 💡 Nota Importante

El mensaje de deprecación sobre `functions.config()` **NO** está causando el error actual. Es solo un aviso para el futuro (marzo 2026). El error real es que las APIs no se habilitaron correctamente.

**La solución más común es simplemente esperar unos minutos y reintentar.**

