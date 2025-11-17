# 🔧 Troubleshooting: Error al Desplegar Firebase Functions

## Error: "An unexpected error has occurred"

Este error suele ocurrir cuando Firebase está habilitando las APIs necesarias. Aquí están las soluciones:

---

## ✅ Solución 1: Esperar y Reintentar

Las APIs pueden tardar unos minutos en habilitarse completamente. Espera 2-3 minutos y vuelve a intentar:

```bash
firebase deploy --only functions
```

---

## ✅ Solución 2: Habilitar APIs Manualmente

Ve a Google Cloud Console y habilita las APIs manualmente:

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Selecciona tu proyecto: **caria-9b633**
3. Ve a **APIs & Services** → **Library**
4. Busca y habilita estas APIs:
   - ✅ **Cloud Functions API**
   - ✅ **Cloud Build API**
   - ✅ **Artifact Registry API**
   - ✅ **Cloud Logging API**

Después de habilitarlas, espera 1-2 minutos y vuelve a intentar el deploy.

---

## ✅ Solución 3: Verificar Facturación

Aunque ya te suscribiste a "pay as you go", verifica:

1. Firebase Console → **Project Settings** → **Usage and billing**
2. Verifica que el plan Blaze esté activo
3. Si no está activo, haz click en **"Upgrade"** o **"Modify plan"**

---

## ✅ Solución 4: Verificar Permisos

Asegúrate de tener los permisos necesarios:

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. **IAM & Admin** → **IAM**
3. Verifica que tu cuenta tenga estos roles:
   - ✅ **Firebase Admin**
   - ✅ **Cloud Functions Admin**
   - ✅ **Service Account User**

---

## ✅ Solución 5: Limpiar y Reintentar

```bash
# Limpiar cache de Firebase
firebase logout
firebase login

# Verificar configuración
firebase use caria-9b633

# Intentar deploy nuevamente
firebase deploy --only functions
```

---

## ✅ Solución 6: Verificar Configuración de Python

Asegúrate de que `firebase.json` esté correcto:

```json
{
  "functions": [
    {
      "source": "functions",
      "codebase": "default",
      "runtime": "python311"
    }
  ]
}
```

Y que `functions/requirements.txt` exista y tenga contenido:

```txt
firebase-functions>=0.1.0
requests>=2.31.0
```

---

## ✅ Solución 7: Usar Emulador Local Primero

Prueba localmente antes de desplegar:

```bash
# Instalar dependencias localmente
cd functions
pip install -r requirements.txt

# Probar con emulador
firebase emulators:start --only functions
```

Si funciona localmente, el problema es con las APIs de Google Cloud.

---

## ✅ Solución 8: Ver Logs Detallados

Ejecuta con más verbosidad:

```bash
firebase deploy --only functions --debug
```

Esto te dará más información sobre qué está fallando exactamente.

---

## ✅ Solución 9: Verificar Variables de Entorno

Asegúrate de que las variables estén configuradas:

```bash
firebase functions:config:get
```

Si no hay nada, configura las variables primero:

```bash
firebase functions:config:set gemini.api_key="TU_API_KEY"
firebase functions:config:set backend.url="http://localhost:8000"
```

---

## 🆘 Si Nada Funciona

1. **Espera 10-15 minutos** - A veces las APIs tardan en propagarse
2. **Intenta desde otro navegador/terminal** - Puede ser un problema de sesión
3. **Contacta soporte de Firebase** - [Firebase Support](https://firebase.google.com/support)

---

## 📋 Checklist de Diagnóstico

Ejecuta estos comandos y comparte los resultados:

```bash
# 1. Verificar proyecto
firebase use

# 2. Verificar login
firebase login:list

# 3. Verificar configuración
firebase functions:config:get

# 4. Verificar APIs (requiere gcloud CLI)
gcloud services list --enabled --project=caria-9b633
```

---

## 💡 Solución Más Común

En el 90% de los casos, el problema se resuelve:
1. Esperando 2-3 minutos
2. Habilitando las APIs manualmente en Google Cloud Console
3. Reintentando el deploy

¡Intenta esto primero!

