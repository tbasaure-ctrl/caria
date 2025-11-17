#!/bin/bash
# Script de setup para Firebase Functions

echo "🚀 Configurando Firebase Functions para Wise Adviser"
echo ""

# Verificar si Firebase CLI está instalado
if ! command -v firebase &> /dev/null; then
    echo "❌ Firebase CLI no está instalado."
    echo "Instala con: npm install -g firebase-tools"
    exit 1
fi

echo "✅ Firebase CLI encontrado"
echo ""

# Login a Firebase
echo "📝 Iniciando sesión en Firebase..."
firebase login

echo ""
echo "🔧 Configurando variables de entorno..."
echo ""
echo "Por favor, ingresa los siguientes valores:"
echo ""

read -p "GEMINI_API_KEY: " GEMINI_KEY
read -p "BACKEND_URL (default: http://localhost:8000): " BACKEND_URL
BACKEND_URL=${BACKEND_URL:-http://localhost:8000}

read -p "¿Quieres configurar Llama como fallback? (y/n): " SETUP_LLAMA

# Configurar Gemini
firebase functions:config:set gemini.api_key="$GEMINI_KEY"
firebase functions:config:set gemini.api_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
firebase functions:config:set backend.url="$BACKEND_URL"

# Configurar Llama si el usuario quiere
if [ "$SETUP_LLAMA" = "y" ] || [ "$SETUP_LLAMA" = "Y" ]; then
    read -p "LLAMA_API_KEY: " LLAMA_KEY
    read -p "LLAMA_API_URL: " LLAMA_URL
    read -p "LLAMA_MODEL_NAME (default: llama-3.1-70b-instruct): " LLAMA_MODEL
    LLAMA_MODEL=${LLAMA_MODEL:-llama-3.1-70b-instruct}
    
    firebase functions:config:set llama.api_key="$LLAMA_KEY"
    firebase functions:config:set llama.api_url="$LLAMA_URL"
    firebase functions:config:set llama.model_name="$LLAMA_MODEL"
fi

echo ""
echo "✅ Configuración completada!"
echo ""
echo "Para desplegar las funciones, ejecuta:"
echo "  firebase deploy --only functions"
echo ""

