# 🔍 Solución: No Veo los Requests en Insomnia

## ✅ Paso 1: Expandir la Colección

Los requests están **dentro** de la colección. Necesitas hacer click para expandirla:

1. En el **sidebar izquierdo**, busca la colección que importaste
2. Busca algo como:
   - "CARIA API"
   - "CARIA API - Simple"
   - "Imported Workspace"
3. **Click en el nombre** de la colección (o en la flecha ▶️ si la hay)
4. Deberías ver los requests aparecer debajo

## 📍 Dónde Buscar los Requests

### En el Sidebar Izquierdo:

```
📁 CARIA API - Simple  ← Click aquí para expandir
   📄 1. Health Check
   📄 2. Register
   📄 3. Login (COPIA EL TOKEN)
   📄 4. Prices - Batch (AGREGA TOKEN)
   ...
```

### Si No Ves Nada:

1. **Busca en "Collections"** en el sidebar izquierdo
2. O busca en el **filtro** arriba del sidebar
3. O usa **Ctrl + P** para buscar "Health Check" o "Login"

## 🎯 Pasos Detallados:

### Opción A: Desde el Sidebar

1. Mira el **sidebar izquierdo**
2. Busca una sección que diga "Collections" o "All Files"
3. Dentro de ahí, busca "CARIA API" o "Imported Workspace"
4. **Click en el nombre** de la colección
5. Los requests deberían aparecer debajo

### Opción B: Desde la Vista de Grid

1. Si estás en la vista de grid (varios cards de colecciones)
2. **Click en el card** que dice "CARIA API" o "Imported Workspace"
3. Esto debería abrir la colección y mostrar los requests

### Opción C: Buscar Directamente

1. Presiona **Ctrl + P** (o click en la barra de búsqueda)
2. Escribe: `Health Check` o `Login`
3. Deberías ver los requests aparecer

## 🔧 Si Aún No Los Ves:

### Verificar que se Importó Correctamente:

1. Ve a **File** → **Import** → **From File**
2. Selecciona `insomnia_collection_simple.json` nuevamente
3. Asegúrate de que diga "Import successful"

### Crear Request Manualmente (Plan B):

Si no encuentras los requests, puedes crear uno manualmente:

1. Click en el botón **"+"** grande (arriba a la izquierda)
2. Selecciona **"HTTP Request"**
3. Configura:
   - **Method**: GET
   - **URL**: `http://localhost:8000/health`
   - **Name**: Health Check
4. Click en **"Send"**

## 📋 Lista de Requests que Deberías Ver:

Si expandiste la colección correctamente, deberías ver:

1. **1. Health Check** - `GET http://localhost:8000/health`
2. **2. Register** - `POST http://localhost:8000/api/auth/register`
3. **3. Login (COPIA EL TOKEN)** - `POST http://localhost:8000/api/auth/login`
4. **4. Prices - Batch (AGREGA TOKEN)** - `POST http://localhost:8000/api/prices/realtime`
5. **5. Holdings - List (AGREGA TOKEN)** - `GET http://localhost:8000/api/holdings`
6. **6. Holdings - Create (AGREGA TOKEN)** - `POST http://localhost:8000/api/holdings`
7. **7. Holdings - With Prices (AGREGA TOKEN)** - `GET http://localhost:8000/api/holdings/with-prices`
8. **8. Valuation - Quick (AGREGA TOKEN)** - `POST http://localhost:8000/api/valuation/AAPL`
9. **9. Regime - Current (SIN TOKEN)** - `GET http://localhost:8000/api/regime/current`

## 💡 Tip Visual:

En Insomnia, la estructura es así:

```
📁 Colección (Click para expandir)
   📄 Request 1
   📄 Request 2
   📄 Request 3
```

Si solo ves la colección pero no los requests, **haz click en la colección** para expandirla.

## 🚀 Prueba Rápida:

1. **Click en cualquier colección** que veas en el sidebar
2. Si se expande y ves requests → ¡Perfecto!
3. Si no pasa nada → Intenta crear un request manualmente (ver arriba)

## 📞 Si Nada Funciona:

Usa el script de Python en su lugar:

```bash
cd services/api
python test_api_connection.py
```

Este script prueba todos los endpoints automáticamente sin necesidad de Insomnia.












