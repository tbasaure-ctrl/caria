# 🚀 Cómo Usar Insomnia - Guía Paso a Paso

## 📍 Dónde Está el Environment en Insomnia

El "Environment" puede estar en diferentes lugares según tu versión:

### Versión Nueva (Insomnia 2023+)
1. **Arriba a la derecha** hay un dropdown que dice "No Environment" o "Base Environment"
2. Click ahí para abrir/editar environments

### Versión Antigua
1. Click en el ícono de **"Manage Environments"** (globo/planeta) en la barra superior
2. O busca en el menú: **"Preferences"** → **"Data"** → **"Manage Environments"**

### Si NO Lo Encuentras
**No te preocupes!** Puedes usar la versión simple sin environments (ver abajo).

---

## ✅ Opción 1: Versión Simple (Sin Environments)

### Paso 1: Importar Colección Simple

1. Abre Insomnia
2. Click en **"+"** o **"Create"** → **"Import"** → **"From File"**
3. Selecciona `insomnia_collection_simple.json`
4. Verás requests numerados del 1 al 9

### Paso 2: Probar Sin Token (Health Check)

1. Abre **"1. Health Check"**
2. Click en **"Send"**
3. Deberías ver una respuesta con el estado de la API

### Paso 3: Hacer Login

1. Abre **"2. Register"** (opcional, solo si es primera vez)
2. Click en **"Send"**
3. Abre **"3. Login (COPIA EL TOKEN)"**
4. Click en **"Send"**
5. **IMPORTANTE**: En la respuesta, busca `access_token` y cópialo
   - Ejemplo: `"access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."`
   - Copia TODO el token (es largo)

### Paso 4: Agregar Token a Otros Requests

Para cada request que dice "(AGREGA TOKEN)":

1. Abre el request (ej: "4. Prices - Batch")
2. Ve al tab **"Header"**
3. Busca el header `Authorization`
4. Reemplaza `PEGA_TU_TOKEN_AQUI` con tu token real
5. Debería quedar: `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
6. Click en **"Send"**

---

## ✅ Opción 2: Usar Script de Python (Más Fácil)

Si Insomnia te da problemas, usa el script de Python:

```bash
cd services/api
python test_api_connection.py
```

Este script:
- ✅ Prueba todos los endpoints automáticamente
- ✅ Maneja el login y tokens automáticamente
- ✅ Muestra resultados con colores
- ✅ No requiere configuración manual

---

## 📸 Guía Visual Paso a Paso

### 1. Importar Colección

```
Insomnia → Click "+" → Import → From File → Selecciona insomnia_collection_simple.json
```

### 2. Ver Requests

Después de importar, verás una lista como:
```
📁 CARIA API - Simple
  📄 1. Health Check
  📄 2. Register
  📄 3. Login (COPIA EL TOKEN)
  📄 4. Prices - Batch (AGREGA TOKEN)
  📄 5. Holdings - List (AGREGA TOKEN)
  ...
```

### 3. Hacer Login

1. Click en **"3. Login (COPIA EL TOKEN)"**
2. Verás:
   - **URL**: `http://localhost:8000/api/auth/login`
   - **Method**: POST
   - **Body**: `username=testuser&password=TestPassword123!`
3. Click en **"Send"** (botón azul arriba a la derecha)
4. Abajo verás la respuesta con el token

### 4. Copiar Token

En la respuesta del login, busca:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6MTcwNDY3MjAwMH0.abc123..."
}
```

Copia TODO el valor de `access_token` (desde `eyJ` hasta el final).

### 5. Agregar Token a un Request

1. Abre **"4. Prices - Batch (AGREGA TOKEN)"**
2. Click en el tab **"Header"** (arriba, junto a Body, Auth, etc.)
3. Verás:
   ```
   Authorization: Bearer PEGA_TU_TOKEN_AQUI
   ```
4. Selecciona `PEGA_TU_TOKEN_AQUI` y reemplázalo con tu token
5. Debería quedar:
   ```
   Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6MTcwNDY3MjAwMH0.abc123...
   ```
6. Click en **"Send"**

---

## 🎯 Orden Recomendado de Pruebas

1. ✅ **1. Health Check** - Verifica que la API esté corriendo
2. ✅ **2. Register** - Crea un usuario (solo primera vez)
3. ✅ **3. Login** - Obtén el token (COPIA EL TOKEN)
4. ✅ **9. Regime - Current** - Prueba sin token
5. ✅ **4. Prices - Batch** - Prueba con token
6. ✅ **5. Holdings - List** - Prueba con token
7. ✅ **6. Holdings - Create** - Crea un holding
8. ✅ **7. Holdings - With Prices** - Ve tu portfolio
9. ✅ **8. Valuation - Quick** - Valua una empresa

---

## 🐛 Problemas Comunes

### "No se pudo conectar"
- Verifica que la API esté corriendo: `python start_api.py`
- Verifica que la URL sea `http://localhost:8000`

### "Unauthorized" (401)
- Verifica que hayas hecho login
- Verifica que el token esté completo (es muy largo)
- Verifica que el header diga `Bearer ` seguido del token (con espacio)

### Token expirado
- Simplemente haz login nuevamente y copia el nuevo token

### No veo el tab "Header"
- Busca tabs como "Headers", "Auth", o un ícono de engranaje
- En algunas versiones está en el lado derecho

---

## 💡 Tip Pro: Guardar Token en un Archivo

1. Después del login, copia el token
2. Pégalo en un archivo de texto llamado `token.txt`
3. Cuando necesites usarlo, solo cópialo desde ahí

---

## 📞 ¿Necesitas Más Ayuda?

Si sigues teniendo problemas:
1. Usa el script de Python: `python test_api_connection.py`
2. Revisa `GUIA_INSOMNIA_SIMPLE.md` para más detalles
3. Verifica que la API esté corriendo en `http://localhost:8000`












