# 🐳 Reconstruir Docker - Guía Rápida

## 🔧 Problema Detectado

- Docker no está corriendo
- Error en `docker-compose.yml` (variable `GEMINI_API_URL_API_URL` incorrecta)
- Backend necesario para: Login, Chat, Valuación, Portfolio Analysis

## ✅ Solución: Reconstruir Docker

### Paso 1: Detener contenedores existentes (si hay)

```powershell
cd C:\key\wise_adviser_cursor_context\notebooks\services
docker-compose down
```

### Paso 2: Reconstruir las imágenes

```powershell
docker-compose build --no-cache
```

### Paso 3: Levantar los servicios

```powershell
docker-compose up -d
```

### Paso 4: Verificar que estén corriendo

```powershell
docker-compose ps
```

Deberías ver:
- `caria_db` (PostgreSQL) - Status: Up
- `caria_api` (FastAPI Backend) - Status: Up

### Paso 5: Ver logs (opcional, para verificar)

```powershell
docker-compose logs -f api
```

Presiona `Ctrl+C` para salir de los logs.

## 🔍 Verificar que el Backend Funciona

Abre en tu navegador:
- Health check: http://localhost:8000/health/live
- Debería responder: `{"status":"ok"}`

## 📝 Notas

- El backend necesita estar corriendo en `http://localhost:8000` para que funcionen:
  - ✅ Login/Register
  - ✅ Chat (WebSocket)
  - ✅ Valuación
  - ✅ Portfolio Analysis
  - ✅ Holdings Management

- Firebase Functions solo maneja el endpoint `/api/analysis/challenge` (Analysis Tool)

## 🆘 Si Hay Problemas

### Error: "Port already in use"
```powershell
# Ver qué está usando el puerto 8000
netstat -ano | findstr :8000

# O cambiar el puerto en docker-compose.yml:
# API_PORT=8001
```

### Error: "Cannot connect to database"
```powershell
# Verificar que PostgreSQL esté corriendo
docker-compose logs postgres
```

### Reconstruir desde cero
```powershell
docker-compose down -v  # Elimina volúmenes también
docker-compose build --no-cache
docker-compose up -d
```

