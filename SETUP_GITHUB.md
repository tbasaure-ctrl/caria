# 🚀 Setup GitHub para Vercel - Guía Rápida

## ✅ Opción A: Con GitHub (Recomendado para CI/CD)

### Paso 1: Crear Repositorio en GitHub

1. Ve a: https://github.com/new
2. Nombre: `wise-adviser` (o el que prefieras)
3. **NO** marques "Initialize with README"
4. Click **"Create repository"**
5. **Copia la URL** que te da (ej: `https://github.com/tu-usuario/wise-adviser.git`)

### Paso 2: Inicializar Git Localmente

```powershell
cd C:\key\wise_adviser_cursor_context\notebooks

# Inicializar Git
git init

# Agregar todos los archivos
git add .

# Primer commit
git commit -m "Initial commit: Wise Adviser with Firebase Functions and Vercel config"

# Agregar remote (reemplaza con tu URL)
git remote add origin https://github.com/TU-USUARIO/TU-REPO.git

# Cambiar a branch main
git branch -M main

# Push inicial
git push -u origin main
```

### Paso 3: Deploy en Vercel

1. Ve a: https://vercel.com/new
2. Click **"Import Git Repository"**
3. Selecciona tu repo
4. Configura:
   - **Root Directory:** `caria_data/caria-app`
   - **Framework:** Vite
5. Variables de entorno:
   - `VITE_API_URL` = `https://tu-backend.com`
6. Click **"Deploy"**

---

## ✅ Opción B: Sin GitHub (Deploy Directo)

**No necesitas GitHub** si usas Vercel CLI:

```powershell
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data\caria-app

# Instalar Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy (primera vez te pedirá configuración)
vercel

# Deploy a producción
vercel --prod
```

**Ventaja:** Más rápido, no necesitas GitHub  
**Desventaja:** No tienes CI/CD automático

---

## 🎯 ¿Cuál Elegir?

### Usa GitHub si:
- ✅ Quieres CI/CD automático (cada push = nuevo deploy)
- ✅ Quieres historial de versiones
- ✅ Trabajas en equipo
- ✅ Quieres revisar cambios antes de deployar

### Usa CLI directo si:
- ✅ Solo quieres deployar rápido
- ✅ No necesitas CI/CD
- ✅ Trabajas solo
- ✅ Quieres probar rápido

---

## 📝 Comandos Git Útiles

```powershell
# Ver estado
git status

# Agregar cambios
git add .

# Commit
git commit -m "Descripción de cambios"

# Push
git push origin main

# Ver remotes
git remote -v

# Cambiar remote URL
git remote set-url origin NUEVA_URL
```

---

## ⚠️ Importante: .gitignore

Ya creé un `.gitignore` en la raíz del proyecto que excluye:
- `node_modules/`
- `.env` files
- `venv/`
- `__pycache__/`
- `dist/`
- Secrets y keys

**Revisa** que no estés subiendo información sensible antes de hacer push.

---

## 🚀 Después de Subir a GitHub

1. Ve a Vercel Dashboard
2. Importa tu repo
3. Configura variables de entorno
4. Deploy!

---

## 💡 Recomendación

**Usa GitHub** - Te da más flexibilidad y CI/CD automático. Cada vez que hagas `git push`, Vercel automáticamente redeployará tu app.

