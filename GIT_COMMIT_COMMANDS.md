# Comandos Git para Commit y Push

## Paso 1: Agregar todos los cambios

```powershell
cd C:\key\wise_adviser_cursor_context\notebooks

# Agregar archivos modificados y nuevos
git add .

# O si prefieres agregar específicamente:
git add backend/
git add caria-lib/
git add frontend/
git add *.md
git add scripts/
git add verify_deployment.py
```

## Paso 2: Verificar qué se va a commitear

```powershell
git status
```

## Paso 3: Hacer commit con mensaje descriptivo

```powershell
git commit -m "Migrate from Google Cloud/Gemini to Railway/Neon/Groq

- Remove all Google Cloud and Gemini dependencies
- Update backend for Railway + Neon PostgreSQL
- Replace Gemini with Groq Llama API
- Update frontend to use Railway API instead of Firebase Functions
- Remove Cloud SQL socket handling, use standard Neon connections
- Update all LLM routes to use Groq only
- Add comprehensive deployment documentation
- Create verification scripts and env.example template"
```

## Paso 4: Push a GitHub

```powershell
git push origin main
```

## Si hay conflictos o necesitas forzar (solo si es necesario)

```powershell
# Ver el estado remoto primero
git fetch origin

# Si hay cambios remotos, hacer pull primero
git pull origin main

# Si todo está bien, hacer push
git push origin main
```

## Comandos completos en una secuencia

```powershell
cd C:\key\wise_adviser_cursor_context\notebooks
git add .
git commit -m "Migrate from Google Cloud/Gemini to Railway/Neon/Groq - Remove all GCP dependencies, update for Railway deployment, replace Gemini with Groq Llama"
git push origin main
```

