@echo off
echo Obteniendo logs del ultimo despliegue...
gcloud run services logs read caria-api --region=us-central1 --limit=200

