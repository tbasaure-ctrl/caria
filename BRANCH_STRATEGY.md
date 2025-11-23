# Git Branch Strategy - Vercel Deployment

## Situación Actual

**Branch actual:** `cursor/configure-railway-deployment-and-api-client-default-e0a6` (feature branch)
**Branch de producción:** `main` (probablemente)

## El Problema

Vercel está configurado para desplegar desde `main` (production branch), pero todos los cambios están en el branch de feature. Por eso Vercel no está desplegando automáticamente.

## Soluciones

### Opción 1: Mergear a Main (Recomendado para Producción)

```bash
# 1. Asegúrate de que todo esté commiteado
git status

# 2. Cambia a main
git checkout main

# 3. Actualiza main
git pull origin main

# 4. Mergea el branch de feature
git merge cursor/configure-railway-deployment-and-api-client-default-e0a6

# 5. Push a main
git push origin main
```

**Resultado:** Vercel detectará el push a `main` y desplegará automáticamente.

### Opción 2: Configurar Vercel para Desplegar desde Feature Branch

1. **Vercel Dashboard → Settings → Git**
2. **Production Branch:** Cambiar temporalmente a `cursor/configure-railway-deployment-and-api-client-default-e0a6`
3. O crear un **Preview Deployment** desde este branch

### Opción 3: Forzar Deployment Manual

1. **Vercel Dashboard → Deployments**
2. **"Create Deployment"**
3. **Branch:** Selecciona `cursor/configure-railway-deployment-and-api-client-default-e0a6`
4. **Deploy**

## Recomendación

**Para producción:** Mergear a `main` es la mejor práctica. Los feature branches son para desarrollo, y `main` es para producción.

**Pasos:**
1. Revisa que todo funcione en el feature branch
2. Mergea a `main`
3. Vercel desplegará automáticamente desde `main`

## Verificar Branch de Producción en Vercel

1. **Vercel Dashboard → Settings → Git**
2. **Production Branch:** Debería mostrar `main` o `master`
3. Esto es lo que Vercel monitorea para deployments automáticos

## Branches Disponibles

- `main` - Branch de producción (Vercel debería desplegar desde aquí)
- `cursor/configure-railway-deployment-and-api-client-default-e0a6` - Feature branch actual
- Otros feature branches

## Próximos Pasos

1. **Decide:** ¿Quieres mergear a main ahora o seguir trabajando en el feature branch?
2. **Si mergeas a main:** Vercel desplegará automáticamente
3. **Si sigues en feature branch:** Usa deployment manual o cambia Production Branch en Vercel
