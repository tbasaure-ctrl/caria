# Pasos Detallados de Reorganización (SIN PÉRDIDA DE CONTENIDO)

## ✅ VERIFICACIÓN PREVIA COMPLETADA

- ✓ `src/` tiene 58 archivos (versión antigua/duplicada)
- ✓ `caria_data/src/` tiene 79 archivos (versión actual usada en producción)
- ✓ Todos los archivos de `src/` también están en `caria_data/src/`
- ✓ Dockerfile usa `caria_data/src/` (versión de producción)
- ✓ Carpeta `backups/` creada

## 📋 PASOS A EJECUTAR

### Paso 1: Backup de Seguridad
```bash
git commit -m "Backup antes de reorganización"
```

### Paso 2: Crear Nueva Estructura
```bash
mkdir -p backend caria-lib frontend configs scripts docs infrastructure deployment backups/src_old
```

### Paso 3: Mover Carpetas Principales
```bash
# Mover backend (sin cambios internos)
mv services/* backend/

# Mover caria-lib (biblioteca core)
mv caria_data/src/caria caria-lib/caria
cp caria_data/requirements.txt caria-lib/requirements.txt

# Mover frontend
mv caria_data/caria-app frontend/caria-app

# Backup de src/ duplicado (antes de eliminar)
mv src backups/src_old/
```

### Paso 4: Consolidar Configuraciones
```bash
# Mover configs de caria_data/ a configs/ (combinar si hay duplicados)
cp -r caria_data/configs/* configs/ 2>/dev/null || true
cp -r configs/* configs/ 2>/dev/null || true  # Consolidar duplicados
```

### Paso 5: Consolidar Scripts
```bash
# Mover scripts de caria_data/ a scripts/
cp -r caria_data/scripts/* scripts/ 2>/dev/null || true
cp -r scripts/* scripts/ 2>/dev/null || true  # Ya existen, consolidar
```

### Paso 6: Organizar Documentación
```bash
# Mover docs sueltos a docs/
mv *.md docs/ 2>/dev/null || true
mv caria_data/*.md docs/ 2>/dev/null || true
mv services/*.md docs/ 2>/dev/null || true
# Excepciones: README.md, REORGANIZACION_*.md (quedan en raíz)
```

### Paso 7: Actualizar Dockerfile
```dockerfile
# Cambiar de:
COPY services/ /app/services/
COPY caria_data/src/ /app/caria_data/src/

# A:
COPY backend/ /app/backend/
COPY caria-lib/ /app/caria-lib/

# Actualizar PYTHONPATH
ENV PYTHONPATH=/app/caria-lib:/app/backend:$PYTHONPATH
```

### Paso 8: Actualizar Imports (si es necesario)
```python
# Los imports de 'caria.*' seguirán funcionando si PYTHONPATH incluye caria-lib/
# Pero verificar rutas relativas en app.py
```

### Paso 9: Actualizar cloudbuild.yaml
```yaml
# Cambiar rutas si es necesario
# Verificar que apunte a backend/Dockerfile
```

### Paso 10: Probar Build Local
```bash
docker build -t caria-api-test -f backend/Dockerfile .
```

### Paso 11: Probar Deployment
```bash
gcloud run deploy caria-api --source . --region us-central1 ...
```

### Paso 12: Verificar Contenido
```bash
# Comparar conteo de archivos antes/después
# Verificar que no se perdió nada importante
```

## ⚠️ ARCHIVOS QUE NO SE MUEVEN

- `data/` - queda como está (datos)
- `models/`, `artifacts/` - datos generados
- `lightning_logs/` - logs de entrenamiento
- `.git/` - repositorio git
- `node_modules/` - dependencias

## 🔄 ESTRUCTURA ANTES → DESPUÉS

### ANTES:
```
notebooks/
├── services/          → backend/
├── caria_data/
│   ├── src/caria/     → caria-lib/caria/
│   ├── caria-app/     → frontend/caria-app/
│   └── configs/       → configs/ (consolidar)
├── src/               → backups/src_old/ (backup)
└── *.md               → docs/ (organizar)
```

### DESPUÉS:
```
notebooks/
├── backend/           (API FastAPI)
├── caria-lib/         (Biblioteca core)
├── frontend/          (React app)
├── configs/           (Configuraciones)
├── scripts/           (Scripts consolidados)
├── docs/              (Documentación)
├── data/              (Sin cambios)
├── backups/
│   └── src_old/      (Backup del src/ duplicado)
└── deployment/        (Cloud Build, GitHub Actions)
```

## ✅ GARANTÍAS

1. ✓ Todo se **mueve**, nada se **elimina** directamente
2. ✓ `src/` duplicado va a `backups/` antes de eliminar
3. ✓ Commits en cada paso importante
4. ✓ Verificación de conteo de archivos antes/después
5. ✓ Build y deployment probados antes de limpiar backups

## 🚀 BENEFICIOS

- ✅ Estructura clara y jerárquica
- ✅ Fácil de navegar y entender
- ✅ Deployment más simple
- ✅ Compatible con Cloud Run
- ✅ Sin pérdida de contenido

