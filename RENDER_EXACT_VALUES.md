# Valores Exactos para Render Dashboard

## En "Verify Settings" o "Build & Deploy" Settings:

### Root Directory
```
(vacío - dejar en blanco)
```
O si no te deja vacío:
```
.
```

### Dockerfile Path  
```
backend/Dockerfile
```

### Docker Build Context Directory
```
.
```

### Docker Command (Start Command)
```
/app/backend/start.sh
```

## Si Render no acepta estos valores:

### Opción 1: Usar valores absolutos desde raíz
Si tu repo tiene estructura `notebooks/backend/Dockerfile`:

**Root Directory:**
```
notebooks
```

**Dockerfile Path:**
```
backend/Dockerfile
```

**Docker Build Context Directory:**
```
notebooks
```

### Opción 2: Si el repo root es directamente notebooks/
Entonces desde la raíz del repo:

**Root Directory:**
```
(vacío)
```

**Dockerfile Path:**
```
backend/Dockerfile
```

**Docker Build Context Directory:**
```
.
```

## Verificación

Después de guardar, los logs deberían mostrar:
- ✅ Cloning repository
- ✅ Building Docker image
- ✅ No debería decir "backend/Dockerfile is missing"


