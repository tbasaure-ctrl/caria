@echo off
echo ===========================================
echo Ver logs del servicio caria-api
echo ===========================================
echo.

gcloud run services logs read caria-api --region=us-central1 --limit=100

echo.
echo ===========================================
echo Si ves errores, busca:
echo - "ModuleNotFoundError" = falta dependencia
echo - "Connection refused" = problema con Cloud SQL
echo - "ImportError" = problema con imports
echo ===========================================
pause

