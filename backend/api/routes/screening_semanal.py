"""
Screening Semanal - Endpoints para calcular y guardar screenings usando CariaScoreEngine
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Optional, List
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Importar nuestros módulos
from api.backend_score import CariaScoreEngine
from api.database_manager import init_db, guardar_resultados

# Cargar .env solo si estamos en local
load_dotenv()

router = APIRouter(prefix="/api/screening", tags=["screening-semanal"])

# Leer variables de entorno (Railway o .env)
API_KEY = os.getenv("FMP_API_KEY")
EMPRESAS_DEFAULT = ["AAPL", "TSLA", "UNH", "NVDA", "AMD", "MSFT"]

# Inicializar base de datos al importar el módulo
init_db()

# Definir el modelo de datos (Schema)
class DesgloseScore(BaseModel):
    Quality: float
    Valuation: float
    Momentum: float
    Catalysts: float
    Risk_Safety: float

class ScreeningItem(BaseModel):
    Ticker: str
    # Usamos Optional[str] = None para que si llega vacío, no dé error
    company_name: Optional[str] = "Empresa Desconocida"
    sector: Optional[str] = "General"
    C_Score: float
    Desglose: DesgloseScore

@router.get("/", response_model=List[ScreeningItem])
def get_screening_live():
    """
    Calcula los scores en tiempo real y devuelve el JSON.
    (No guarda en base de datos, solo muestra).
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Falta configurar FMP_API_KEY")

    engine = CariaScoreEngine(API_KEY, EMPRESAS_DEFAULT)
    df = engine.calculate_scores()
    # Asegurarse de que devuelve una lista de diccionarios
    return df.to_dict(orient="records")

@router.post("/run-weekly-job")
def run_weekly_job(background_tasks: BackgroundTasks):
    """
    Endpoint para forzar el guardado en base de datos.
    Se ejecuta en segundo plano para no congelar la API.
    """
    background_tasks.add_task(ejecutar_y_guardar)
    return {"message": "Proceso de screening y guardado iniciado en segundo plano."}

def ejecutar_y_guardar():
    print("⏳ Iniciando Job Semanal...")
    engine = CariaScoreEngine(API_KEY, EMPRESAS_DEFAULT)
    df = engine.calculate_scores()
    resultado = guardar_resultados(df)
    print(f"✅ Job finalizado: {resultado}")
