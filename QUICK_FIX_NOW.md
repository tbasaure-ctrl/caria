# 🚀 Quick Fix - Migrar a Google Cloud Platform

## ✅ Nueva Estrategia: Google Cloud Platform

Ya que Railway está dando problemas y vamos a usar Gemini, migramos a **Google Cloud Platform**:

- ✅ **Cloud Run** - Serverless, escala automáticamente
- ✅ **Cloud SQL** - PostgreSQL con pgvector nativo
- ✅ **Integración Gemini** - Mismo ecosistema Google
- ✅ **Más fácil de mantener** - Menos configuración

## 🚀 Inicio Rápido (3 pasos)

### Paso 1: Setup Inicial (5 minutos)

```bash
# Instalar Google Cloud SDK si no lo tienes
# https://cloud.google.com/sdk/docs/install

# Ejecutar script de setup interactivo
./setup-gcp.sh
```

Este script te guiará para:
- Crear/seleccionar proyecto GCP
- Habilitar APIs necesarias
- Crear Cloud SQL (PostgreSQL)
- Configurar Secret Manager (Gemini API Key)

### Paso 2: Desplegar Backend (2 minutos)

```bash
# Configurar variables (si no las configuraste en setup)
export CLOUDSQL_INSTANCE=proyecto:region:caria-db
export DATABASE_URL=postgresql://postgres:PASSWORD@/caria?host=/cloudsql/proyecto:region:caria-db

# Desplegar
./deploy-gcp.sh
```

### Paso 3: Actualizar Frontend (1 minuto)

1. Ve a Vercel Dashboard → Tu proyecto → Settings → Environment Variables
2. Actualiza `VITE_API_URL` con la URL de Cloud Run que te dio el script
3. Redeploy

## 📋 Guía Completa

Ver `GCP_MIGRATION_GUIDE.md` para instrucciones detalladas.

## 🔧 Archivos Creados

- ✅ `GCP_MIGRATION_GUIDE.md` - Guía completa de migración
- ✅ `cloudbuild.yaml` - Configuración CI/CD automática
- ✅ `setup-gcp.sh` - Script de setup inicial
- ✅ `deploy-gcp.sh` - Script de despliegue rápido

## 💰 Costos

- **Cloud Run**: ~$0.40/millón requests (muy barato)
- **Cloud SQL**: ~$7.50/mes (db-f1-micro)
- **Total**: ~$10-15/mes para empezar

## ✅ Ventajas vs Railway

1. ✅ Mejor integración con Gemini
2. ✅ Cloud SQL con pgvector nativo (más fácil)
3. ✅ Secret Manager integrado
4. ✅ Escala automáticamente a 0 cuando no hay tráfico
5. ✅ Logs y monitoreo mejores

## 🎯 Próximos Pasos

1. Ejecuta `./setup-gcp.sh`
2. Ejecuta `./deploy-gcp.sh`
3. Actualiza Vercel con la nueva URL
4. Prueba login, chat, valuation
5. ¡Listo! 🎉

---

**Nota**: Los scripts están en formato bash. En Windows, puedes usar Git Bash o WSL, o ejecutar los comandos manualmente siguiendo `GCP_MIGRATION_GUIDE.md`.
