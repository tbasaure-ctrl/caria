# ✅ Checklist Pre-Deploy a Vercel

Antes de desplegar, verifica estos puntos:

## 🔧 Configuración del Proyecto

- [ ] `vercel.json` creado y configurado
- [ ] `package.json` tiene script `build`
- [ ] `.vercelignore` configurado
- [ ] Variables de entorno identificadas

## 🌐 Backend

- [ ] Backend FastAPI está accesible públicamente (no localhost)
- [ ] CORS configurado para permitir dominio de Vercel
- [ ] Variables de entorno del backend configuradas
- [ ] PostgreSQL accesible desde el backend (si está en la nube)

## 🔥 Firebase

- [ ] Firebase Functions desplegadas y funcionando
- [ ] URLs de Firebase Functions correctas en `firebaseFunctionsService.ts`
- [ ] Firebase Auth configurado (si lo usas)

## 📝 Variables de Entorno Necesarias

### En Vercel Dashboard:

- [ ] `VITE_API_URL` configurada
  - Desarrollo: `http://localhost:8000`
  - Producción: `https://tu-backend.com`

### Opcionales:

- [ ] `VITE_GEMINI_API_KEY` (si lo usas directamente desde frontend)

## 🧪 Testing Local

Antes de deployar, prueba localmente:

- [ ] `npm run build` funciona sin errores
- [ ] `npm run preview` muestra la app correctamente
- [ ] Login funciona
- [ ] Analysis Tool funciona (Firebase Functions)
- [ ] Chat funciona (WebSocket)
- [ ] Portfolio Analytics funciona

## 📦 Build

- [ ] No hay errores de TypeScript
- [ ] No hay warnings críticos
- [ ] El build genera `dist/` correctamente

## 🔗 URLs a Verificar Después del Deploy

- [ ] Frontend carga correctamente
- [ ] Login funciona
- [ ] Analysis Tool funciona
- [ ] Chat funciona
- [ ] Portfolio Analytics funciona
- [ ] Valuación funciona

---

## 🚀 Comando de Deploy

Una vez que todo esté listo:

```bash
cd caria_data/caria-app
vercel --prod
```

O usa el script:
```powershell
.\deploy-vercel.ps1
```

---

## 🆘 Si Algo Falla

1. Revisa los logs en Vercel Dashboard
2. Verifica variables de entorno
3. Verifica que el backend esté accesible
4. Revisa CORS en el backend
5. Verifica que Firebase Functions estén desplegadas

