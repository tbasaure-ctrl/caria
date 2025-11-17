@echo off
echo Obteniendo logs del servicio...
gcloud run services logs read caria-api --region=us-central1 --limit=200

