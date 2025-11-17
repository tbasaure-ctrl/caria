#!/bin/bash
# Script de deploy a Vercel para Linux/Mac

echo "🚀 Deploy a Vercel - Wise Adviser"
echo ""

# Verificar si Vercel CLI está instalado
if ! command -v vercel &> /dev/null; then
    echo "📦 Instalando Vercel CLI..."
    npm install -g vercel
    if [ $? -ne 0 ]; then
        echo "❌ Error instalando Vercel CLI"
        exit 1
    fi
fi

echo "✅ Vercel CLI encontrado"
echo ""

# Verificar si está logueado
echo "🔐 Verificando login..."
vercel whoami &> /dev/null
if [ $? -ne 0 ]; then
    echo "📝 Necesitas hacer login..."
    vercel login
fi

echo ""
echo "📋 Configuración:"
echo "  - Root Directory: caria_data/caria-app"
echo "  - Framework: Vite"
echo "  - Build Command: npm run build"
echo ""

# Preguntar por variables de entorno
echo "🔧 Variables de Entorno:"
read -p "VITE_API_URL (default: http://localhost:8000): " api_url
api_url=${api_url:-http://localhost:8000}

echo ""
echo "🚀 Iniciando deploy..."
echo ""

# Deploy
vercel --prod

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deploy completado!"
    echo ""
    echo "📝 No olvides configurar las variables de entorno en Vercel Dashboard:"
    echo "   - VITE_API_URL = $api_url"
    echo ""
    echo "   Ve a: https://vercel.com/dashboard → Tu Proyecto → Settings → Environment Variables"
else
    echo ""
    echo "❌ Error en el deploy. Revisa los logs arriba."
fi

