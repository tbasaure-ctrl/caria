# 📦 Guía: Subir Proyecto a GitHub

## 🎯 Opción 1: Si YA tienes un repositorio Git

### Verificar estado actual:
```powershell
cd C:\key\wise_adviser_cursor_context\notebooks
git status
git remote -v
```

### Si ya tienes remote configurado:
```powershell
# Agregar todos los cambios
git add .

# Commit
git commit -m "Add Firebase Functions and Vercel configuration"

# Push
git push origin main
# o
git push origin master
```

---

## 🎯 Opción 2: Crear nuevo repositorio en GitHub

### Paso 1: Crear repo en GitHub

1. Ve a: https://github.com/new
2. Nombre: `wise-adviser` (o el que prefieras)
3. Descripción: "Wise Adviser - Investment Analysis Platform"
4. **NO** marques "Initialize with README" (ya tienes archivos)
5. Click **"Create repository"**

### Paso 2: Conectar tu proyecto local

```powershell
cd C:\key\wise_adviser_cursor_context\notebooks

# Si NO tienes Git inicializado:
git init
git add .
git commit -m "Initial commit: Wise Adviser project"

# Agregar remote (reemplaza USERNAME y REPO_NAME)
git remote add origin https://github.com/USERNAME/REPO_NAME.git

# O si prefieres SSH:
# git remote add origin git@github.com:USERNAME/REPO_NAME.git

# Push
git branch -M main
git push -u origin main
```

---

## 🎯 Opción 3: Usar Vercel CLI SIN GitHub

**No necesitas GitHub** si usas Vercel CLI directamente:

```powershell
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data\caria-app
npm install -g vercel
vercel login
vercel --prod
```

Esto despliega directamente desde tu máquina local.

---

## 📝 .gitignore Recomendado

Asegúrate de tener un `.gitignore` adecuado. Crea o verifica:

```gitignore
# Dependencies
node_modules/
venv/
__pycache__/
*.pyc

# Environment
.env
.env.local
.env.*.local

# Build
dist/
build/
*.log

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Firebase
.firebase/
firebase-debug.log

# Vercel
.vercel/

# Docker
*.env
docker-compose.override.yml
```

---

## ✅ Checklist

- [ ] Repositorio creado en GitHub (si usas Dashboard)
- [ ] Git inicializado localmente
- [ ] `.gitignore` configurado
- [ ] Cambios commiteados
- [ ] Push a GitHub realizado
- [ ] Repositorio público o privado según prefieras

---

## 🚀 Después de Subir a GitHub

1. Ve a Vercel Dashboard
2. Click "Import Git Repository"
3. Selecciona tu repo
4. Configura Root Directory: `caria_data/caria-app`
5. Deploy!

