"""API FastAPI para Google AI Studio."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Import valuation module
from .valuation import ValuationAnalyzer
from .routes import simulation, watchlist, liquidity


# Setup
app = FastAPI(title="Caria API", version="1.0.0")

# CORS para Google AI Studio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from caria.backend.api.routes import liquidity, topology, screener

app.include_router(simulation.router)
app.include_router(watchlist.router)
app.include_router(liquidity.router)
app.include_router(topology.router)
app.include_router(screener.router)

quality_model = None
valuation_model = None
momentum_model = None
valuation_analyzer = None

@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global quality_model, valuation_model, momentum_model, valuation_analyzer

    try:
        quality_model = joblib.load(MODELS_DIR / "quality_model.pkl")
        valuation_model = joblib.load(MODELS_DIR / "valuation_model.pkl")
        momentum_model = joblib.load(MODELS_DIR / "momentum_model.pkl")
        valuation_analyzer = ValuationAnalyzer()
        valuation_analyzer = ValuationAnalyzer()
        
        # Initialize RAG components
        from caria.config.settings import Settings
        from caria.embeddings.generator import EmbeddingGenerator
        from caria.retrieval.vector_store import VectorStore
        
        settings = Settings()
        app.state.settings = settings
        app.state.embedding_generator = EmbeddingGenerator(settings)
        app.state.vector_store = VectorStore.from_settings(settings)
        
        print("[OK] Models, valuation analyzer, and RAG components loaded successfully")
    except Exception as e:
        print(f"[WARNING] Error loading models: {e}")


# Request/Response models
class AnalysisRequest(BaseModel):
    ticker: str
    user_query: str = ""


class BiasDetection(BaseModel):
    bias_type: str
    confidence: float
    explanation: str


class AnalysisResponse(BaseModel):
    ticker: str
    moat_analysis: str
    valuation_context: str
    momentum_signal: str
    detected_biases: list[BiasDetection]
    historical_context: str


# Endpoints
@app.get("/")
async def root():
    return {
        "service": "Caria API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/analyze", "/health"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    models_loaded = all([quality_model, valuation_model, momentum_model])
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_ticker(request: AnalysisRequest):
    """
    Analiza un ticker y retorna insights cualitativos.

    NO retorna scores numéricos.
    SÍ retorna análisis cualitativo + detección de sesgos.
    """

    ticker = request.ticker.upper()

    # Detect biases from user query
    detected_biases = detect_biases_in_query(request.user_query)

    # Get valuation analysis
    valuation_result = valuation_analyzer.analyze(
        ticker=ticker,
        current_price=None,  # Auto-extracted from historical data
        fcf_per_share=None,  # Auto-extracted from historical data
    )

    # Generate response with real analysis
    response = AnalysisResponse(
        ticker=ticker,
        moat_analysis=_generate_moat_analysis(ticker),
        valuation_context=_format_valuation_context(valuation_result),
        momentum_signal=_generate_momentum_signal(ticker),
        detected_biases=detected_biases,
        historical_context=_generate_historical_context(ticker, valuation_result)
    )

    return response


def _generate_moat_analysis(ticker: str) -> str:
    """Genera análisis de moat basado en datos disponibles."""
    # TODO: Implementar análisis real de moat basado en:
    # - ROIC histórico
    # - Márgenes vs competencia
    # - Network effects, switching costs, etc.
    return f"{ticker}: Análisis de moat (basado en ROIC, márgenes, cuota de mercado)"


def _format_valuation_context(valuation_result: dict) -> str:
    """Formatea el contexto de valuación de forma cualitativa."""
    parts = []

    if valuation_result.get("multiples_context"):
        parts.append(valuation_result["multiples_context"])

    if valuation_result.get("dcf_range"):
        parts.append(valuation_result["dcf_range"])

    if valuation_result.get("relative_valuation"):
        parts.append(valuation_result["relative_valuation"])

    return " | ".join(parts) if parts else "Contexto de valuación no disponible"


def _generate_momentum_signal(ticker: str) -> str:
    """Genera señal de momentum basada en indicadores técnicos."""
    # TODO: Usar momentum_model para generar señal
    return f"{ticker}: Momentum neutral (basado en RSI, MACD, SMAs)"


def _generate_historical_context(ticker: str, valuation_result: dict) -> str:
    """Genera contexto histórico relevante."""
    context_parts = []

    # Agregar drivers clave
    if valuation_result.get("key_drivers"):
        context_parts.append("Drivers: " + ", ".join(valuation_result["key_drivers"]))

    # Agregar riesgos
    if valuation_result.get("risks"):
        context_parts.append("Riesgos: " + ", ".join(valuation_result["risks"]))

    return " | ".join(context_parts) if context_parts else "Contexto histórico en progreso"


def detect_biases_in_query(query: str) -> list[BiasDetection]:
    """Detect cognitive biases in user query."""

    biases = []
    query_lower = query.lower()

    # Anchoring
    if any(word in query_lower for word in ["compré a", "estaba a", "volver a", "recuperar"]):
        biases.append(BiasDetection(
            bias_type="anchoring",
            confidence=0.8,
            explanation="Detecté referencia a precio anterior. Tu precio de entrada es irrelevante para valor intrínseco HOY."
        ))

    # Social proof / Herd mentality
    if any(word in query_lower for word in ["todos", "reddit", "twitter", "todo el mundo"]):
        biases.append(BiasDetection(
            bias_type="social_proof",
            confidence=0.9,
            explanation="Detecté influencia social. ¿Decisión propia o siguiendo al rebaño?"
        ))

    # FOMO
    if any(word in query_lower for word in ["rápido", "antes de que", "subiendo mucho", "no quiero perder"]):
        biases.append(BiasDetection(
            bias_type="fomo",
            confidence=0.85,
            explanation="Detecté urgencia. Las mejores inversiones NO requieren prisa."
        ))

    # Loss aversion
    if any(word in query_lower for word in ["no vender", "esperar", "pérdida", "recuperar"]):
        biases.append(BiasDetection(
            bias_type="loss_aversion",
            confidence=0.75,
            explanation="Detecté aversión a pérdida. ¿Mantienes por análisis o por evitar dolor?"
        ))

    # Recency bias
    if any(word in query_lower for word in ["últimamente", "siempre", "nunca", "últimos años"]):
        biases.append(BiasDetection(
            bias_type="recency",
            confidence=0.7,
            explanation="Detecté proyección de tendencias recientes. Historia más larga muestra cycles."
        ))

    return biases


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
