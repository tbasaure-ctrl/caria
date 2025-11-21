# Comandos para Push a GitHub - Nueva Estructura

## ✅ Estado Actual
- Estructura reorganizada: `backend/`, `caria-lib/`, `frontend/`
- Cambios commiteados localmente
- Listo para push

## 📤 Comandos para Push

### Opción 1: Push Simple (si todo está commiteado)
```bash
git push origin main
```

### Opción 2: Verificar y Push (recomendado)
```bash
# 1. Ver estado actual
git status

# 2. Ver commits pendientes
git log --oneline origin/main..HEAD

# 3. Verificar que todo está commiteado
git status

# 4. Hacer push
git push origin main
```

### Opción 3: Si necesitas commitear cambios adicionales
```bash
# 1. Ver qué falta
git status

# 2. Agregar todo
git add -A

# 3. Commit con mensaje descriptivo
git commit -m "Reorganización completa: backend/, caria-lib/, frontend/ - Estructura clara para Cloud Run"

# 4. Push
git push origin main
```

### Opción 4: Si el push falla por tamaño (usar buffer más grande)
```bash
# Configurar buffer más grande
git config http.postBuffer 524288000

# Intentar push
git push origin main
```

### Opción 5: Push forzado con lease (solo si es necesario)
```bash
# ⚠️ SOLO usar si sabes lo que haces
# Esto sobrescribe el remoto, pero verifica que no haya cambios remotos importantes
git push origin main --force-with-lease
```

## 🔍 Verificar Push Exitoso

Después del push, verifica en GitHub:
1. Ve a tu repositorio en GitHub
2. Verifica que aparecen los commits nuevos
3. Verifica que la estructura nueva (`backend/`, `caria-lib/`, `frontend/`) está en GitHub

```bash
# Verificar que el remoto está actualizado
git fetch origin
git log --oneline origin/main -10
```

## 📝 Resumen de Cambios

La nueva estructura incluye:
- ✅ `backend/` - API FastAPI (antes `services/`)
- ✅ `caria-lib/` - Biblioteca core (antes `caria_data/src/caria/`)
- ✅ `frontend/` - React app (antes `caria_data/caria-app/`)
- ✅ `backups/src_old/` - Backup del `src/` duplicado
- ✅ Dockerfile actualizado con nuevas rutas
- ✅ cloudbuild.yaml actualizado
- ✅ start.sh actualizado






