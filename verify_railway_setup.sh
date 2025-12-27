#!/bin/bash
# Script para verificar la configuración de Railway antes de desplegar

echo "🔍 Verificando configuración de Railway..."
echo ""

# Verificar que railway.json existe
if [ -f "railway.json" ]; then
    echo "✅ railway.json encontrado"
else
    echo "❌ railway.json NO encontrado"
    exit 1
fi

# Verificar que el Dockerfile existe
if [ -f "backend/Dockerfile" ]; then
    echo "✅ backend/Dockerfile encontrado"
else
    echo "❌ backend/Dockerfile NO encontrado"
    exit 1
fi

# Verificar que start.sh existe
if [ -f "backend/start.sh" ]; then
    echo "✅ backend/start.sh encontrado"
else
    echo "❌ backend/start.sh NO encontrado"
    exit 1
fi

# Verificar que start.sh es ejecutable
if [ -x "backend/start.sh" ]; then
    echo "✅ backend/start.sh es ejecutable"
else
    echo "⚠️  backend/start.sh NO es ejecutable, corrigiendo..."
    chmod +x backend/start.sh
    echo "✅ Permisos corregidos"
fi

# Verificar que requirements.txt existe
if [ -f "backend/api/requirements.txt" ]; then
    echo "✅ backend/api/requirements.txt encontrado"
else
    echo "❌ backend/api/requirements.txt NO encontrado"
    exit 1
fi

# Verificar que caria-lib existe
if [ -d "caria-lib" ]; then
    echo "✅ caria-lib/ encontrado"
else
    echo "❌ caria-lib/ NO encontrado"
    exit 1
fi

# Verificar que el endpoint /health existe en app.py
if grep -q "/health" backend/api/app.py 2>/dev/null || grep -q "health" backend/api/app.py 2>/dev/null; then
    echo "✅ Endpoint /health encontrado en app.py"
else
    echo "⚠️  Endpoint /health NO encontrado en app.py"
    echo "   Railway está configurado para usar /health como healthcheck"
    echo "   Necesitas agregar este endpoint en backend/api/app.py"
fi

echo ""
echo "📋 Resumen de configuración Railway:"
echo "   - Root Directory: Debe estar VACÍO en Railway Settings"
echo "   - Dockerfile Path: backend/Dockerfile"
echo "   - Start Command: /app/backend/start.sh"
echo "   - Health Check: /health"
echo ""
echo "✅ Verificación completada. Si todos los checks pasaron, puedes desplegar en Railway."

