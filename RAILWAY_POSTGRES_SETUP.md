# 🗄️ Configurar PostgreSQL en Railway

## Pasos para agregar PostgreSQL

1. **Ve a tu proyecto en Railway**
   - URL: https://railway.com/project/ea07210f-237a-40a8-bfcd-fced27730792?environmentId=aee9dd37-42e3-4d36-8da4-eee4b9a07feb

2. **Agregar servicio PostgreSQL**
   - Click en el botón **"Create"** (arriba a la derecha del canvas)
   - Selecciona **"Database"** → **"PostgreSQL"**
   - Railway creará automáticamente el servicio PostgreSQL

3. **Verificar DATABASE_URL**
   - Railway automáticamente crea la variable `DATABASE_URL` cuando agregas PostgreSQL
   - Esta variable se compartirá automáticamente con el servicio "caria"
   - El código ya está configurado para usar `DATABASE_URL` automáticamente

4. **Verificar que el servicio caria tenga acceso**
   - Ve a Variables del servicio "caria"
   - Deberías ver `DATABASE_URL` listada (Railway la comparte automáticamente)
   - Si no aparece, puedes agregarla manualmente desde el servicio PostgreSQL

## Verificación

Una vez configurado PostgreSQL:
- El backend debería poder conectarse automáticamente
- No necesitas configurar variables individuales (POSTGRES_HOST, etc.)
- El código usa `DATABASE_URL` primero, y si no está disponible, usa variables individuales

## Estado actual del código

✅ `get_db_connection()` ya está modificado para usar `DATABASE_URL`
✅ Si `DATABASE_URL` está disponible, la usa automáticamente
✅ Si no está disponible, usa variables individuales como fallback

