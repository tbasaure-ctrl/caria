# Plan de Reorganización del Proyecto

## Análisis Actual

### Problemas Detectados:
1. **Duplicación de código:**
   - `src/caria/` (58 archivos) - versión antigua/duplicada
   - `caria_data/src/caria/` (79 archivos) - versión actual usada en producción

2. **Estructura dispersa:**
   - Archivos .md, .bat, .txt sueltos en la raíz (100+ archivos)
   - Múltiples carpetas con contenido similar (configs, pipelines, scripts)
   - Frontend mezclado con backend

3. **Problemas de deployment:**
   - Dockerfile usa rutas relativas complejas
   - Imports dependen de PYTHONPATH configurado manualmente
   - Estructura confusa para Cloud Build

## Estructura Propuesta (Jerárquica y Clara)

```
notebooks/
├── backend/                    # API y servicios backend
│   ├── api/                   # FastAPI app
│   │   ├── app.py
│   │   ├── dependencies.py
│   │   ├── routes/
│   │   ├── domains/
│   │   └── services/
│   ├── workers/               # Background workers
│   ├── Dockerfile
│   ├── requirements.txt
│   └── start.sh
│
├── caria-lib/                 # Biblioteca core de Caria
│   ├── caria/                 # Código fuente (de caria_data/src/caria)
│   │   ├── config/
│   │   ├── models/
│   │   ├── services/
│   │   ├── pipelines/
│   │   └── ...
│   ├── requirements.txt
│   └── setup.py
│
├── frontend/                  # Frontend React/Vite
│   ├── caria-app/            # (de caria_data/caria-app)
│   │   ├── src/
│   │   ├── components/
│   │   ├── package.json
│   │   └── vite.config.ts
│   └── README.md
│
├── configs/                   # Configuraciones centralizadas
│   ├── environments/
│   ├── pipelines/
│   └── training/
│
├── scripts/                   # Scripts de utilidades
│   ├── data/
│   ├── training/
│   └── maintenance/
│
├── data/                      # Datos (sin cambios)
│   ├── raw/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── infrastructure/            # Infraestructura
│   ├── docker/
│   ├── k8s/
│   └── terraform/
│
├── deployment/                # Archivos de deployment
│   ├── cloudbuild.yaml
│   ├── .github/
│   │   └── workflows/
│   └── vercel.json
│
├── docs/                      # Documentación
│   ├── architecture.md
│   ├── deployment.md
│   └── ...
│
├── backups/                   # Archivos temporales (a limpiar después)
│   └── src_old/              # Backup de src/ duplicado
│
├── .gcloudignore
├── .gitignore
├── README.md
└── docker-compose.yml
```

## Pasos de Migración (SIN PÉRDIDA DE CONTENIDO)

1. **Backup completo** (git commit antes de empezar)
2. **Mover caria_data/src/caria/ → caria-lib/caria/**
3. **Mover services/ → backend/**
4. **Mover caria_data/caria-app/ → frontend/caria-app/**
5. **Consolidar configs, scripts, pipelines**
6. **Mover archivos sueltos a carpetas apropiadas**
7. **Backup de src/ duplicado → backups/src_old/**
8. **Actualizar Dockerfile con nueva estructura**
9. **Actualizar imports y referencias**
10. **Actualizar cloudbuild.yaml y GitHub Actions**
11. **Probar build y deployment**
12. **Limpiar backups después de verificar**

## Ventajas

✅ Estructura clara y jerárquica
✅ Separación frontend/backend/lib
✅ Deployment más simple
✅ Dockerfile más limpio
✅ Fácil navegación
✅ Compatible con Cloud Run
✅ Sin pérdida de contenido

## Riesgos

⚠️ Requiere actualizar imports
⚠️ Requiere actualizar rutas en Dockerfile
⚠️ Requiere probar deployment después

