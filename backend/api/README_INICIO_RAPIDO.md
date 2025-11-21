# Inicio Rápido - API de CARIA

## 🚀 Iniciar la API (3 pasos)

### 1. Configurar Variables de Entorno

```powershell
# Opción A: Script automático
python setup_env.py

# Opción B: Manual - crear archivo .env
# (Ver SETUP_ENV.md para detalles)
```

### 2. Ejecutar Migración (si es primera vez)

```powershell
# Configurar contraseña de PostgreSQL
$env:POSTGRES_PASSWORD='tu_password'

# Ejecutar migración
python run_migration.py
```

### 3. Iniciar la API

```powershell
# Opción A: Script recomendado (carga .env automáticamente)
python start_api.py

# Opción B: Uvicorn directo
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## ✅ Verificar que Funciona

Abre en tu navegador:
- **API**: http://localhost:8000
- **Documentación**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🐛 Problemas Comunes

### Error: "No module named 'api'"
**Solución**: Asegúrate de estar en el directorio `services/api`:
```powershell
cd C:\key\wise_adviser_cursor_context\notebooks\services\api
python start_api.py
```

### Error: "FMP_API_KEY no configurado"
**Solución**: Configura la variable:
```powershell
$env:FMP_API_KEY='your-fmp-api-key-here'
# O crea archivo .env con esa variable
```

### Error: "connection to server failed"
**Solución**: Verifica que PostgreSQL esté corriendo y las credenciales sean correctas.

## 📚 Documentación Completa

- `GUIA_LANZAMIENTO.md` - Guía completa de lanzamiento
- `SETUP_ENV.md` - Configuración de variables de entorno
- `GUIA_UI_WEBSOCKETS.md` - Cómo conectar la UI

