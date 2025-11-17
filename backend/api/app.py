"""Servicio FastAPI para exponer reportes y análisis de Caria."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

# Configurar paths antes de importar módulos de caria
# Este archivo está en backend/api/, necesitamos encontrar caria-lib/
CURRENT_FILE = Path(__file__).resolve()
API_DIR = CURRENT_FILE.parent  # backend/api/
BACKEND_DIR = API_DIR.parent   # backend/

# Cargar variables de entorno desde .env si existe (antes de cualquier otra cosa)
ENV_FILE = API_DIR / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if key.strip() not in os.environ:  # No sobrescribir si ya está configurado
                os.environ[key.strip()] = value.strip()

# Agregar backend/ al path para que Python encuentre el módulo 'api'
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Subir desde backend/api/ hasta notebooks/, luego entrar a caria-lib/
# Nota: PYTHONPATH ya incluye /app/caria-lib en producción, pero para desarrollo local:
CARIA_LIB = CURRENT_FILE.parent.parent.parent / "caria-lib"
if CARIA_LIB.exists() and str(CARIA_LIB) not in sys.path:
    sys.path.insert(0, str(CARIA_LIB))

from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

LOGGER = logging.getLogger("caria.api")

from caria.config.settings import Settings
from caria.embeddings.generator import EmbeddingGenerator
from caria.retrieval.retrievers import Retriever
from caria.retrieval.vector_store import VectorStore

# Modelo legacy (opcional, deprecated)
try:
    from caria.models.training.inference import load_model_bundle
except ImportError:
    LOGGER.warning("Modelo legacy no disponible (pytorch_lightning no instalado). Funcionalidades de predicción legacy deshabilitadas.")
    load_model_bundle = None

# Domain routers per audit document (4.1): Modular monolith architecture
from api.domains.identity import identity_router
from api.domains.portfolio import portfolio_router
from api.domains.social import social_router
from api.domains.analysis import analysis_router as analysis_domain_router
from api.domains.market_data import market_data_router

# Legacy routes (kept for backward compatibility)
# These are now organized under domain routers above
from api.routes.analysis import router as analysis_router
from api.routes.auth import router as auth_router
from api.routes.regime import router as regime_router
from api.routes.factors import router as factors_router
from api.routes.valuation import router as valuation_router
from api.routes.prices import router as prices_router
from api.routes.holdings import router as holdings_router
from api.routes.chat import router as chat_router
from api.routes.community import router as community_router
from api.routes.portfolio_analytics import router as portfolio_analytics_router
from api.routes.monte_carlo import router as monte_carlo_router
from api.routes.model_validation import router as model_validation_router
from api.routes.tactical_allocation import router as tactical_allocation_router
from api.routes.ux_tracking import router as ux_tracking_router

# WebSocket support per audit document
from api.websocket_chat import sio
from socketio import ASGIApp

from api.db_bootstrap import run_bootstrap_tasks
from caria.services.regime_service import RegimeService
from caria.services.factor_service import FactorService
from caria.services.valuation_service import ValuationService


def _load_settings() -> Settings:
    # Determinar path de configuración
    # Si está en env var, usarlo; si no, buscar en caria_data/configs/
    default_config = os.getenv("CARIA_SETTINGS_PATH")
    if not default_config:
        # Desde backend/api/, subir hasta notebooks/, luego caria_data/configs/ (o configs/)
        # Intentar primero en configs/ (estructura nueva), luego caria_data/configs/ (legacy)
        configs_path = CURRENT_FILE.parent.parent.parent / "configs" / "base.yaml"
        if not configs_path.exists():
            configs_path = CURRENT_FILE.parent.parent.parent / "caria_data" / "configs" / "base.yaml"
        caria_data_configs = configs_path
        if caria_data_configs.exists():
            default_config = str(caria_data_configs)
        else:
            default_config = "configs/base.yaml"  # Fallback
    
    settings_path = Path(default_config)
    LOGGER.info("Cargando configuración para API desde %s", settings_path)
    return Settings.from_yaml(settings_path)


def _init_state(app: FastAPI) -> None:
    settings = _load_settings()
    app.state.settings = settings

    # Ensure database schema and default user are in place
    run_bootstrap_tasks()

    # Inicializar RAG (opcional - puede fallar si PostgreSQL no está disponible)
    vector_store = retriever = embedder = None
    try:
        vector_store = VectorStore.from_settings(settings)
        retriever = Retriever(vector_store=vector_store)
        embedder = EmbeddingGenerator(settings=settings)
        LOGGER.info("Stack RAG inicializado correctamente")
    except UnicodeDecodeError as exc:
        LOGGER.warning(
            "Error de encoding al conectar a PostgreSQL. "
            "Esto puede ser causado por caracteres especiales en la connection string. "
            "RAG deshabilitado. Para habilitarlo, verifica las variables de entorno: "
            "POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB. Error: %s",
            exc,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "No se pudo inicializar el stack RAG (esto es opcional y no bloquea la API): %s",
            exc,
        )

    app.state.vector_store = vector_store
    app.state.retriever = retriever
    app.state.embedder = embedder

    # Inicializar servicio de régimen HMM
    try:
        regime_service = RegimeService(settings)
        app.state.regime_service = regime_service
        if regime_service.is_available():
            LOGGER.info("Servicio de régimen HMM inicializado")
        else:
            LOGGER.warning("Servicio de régimen HMM no disponible (modelo no encontrado)")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Error inicializando servicio de régimen: %s", exc)
        app.state.regime_service = None

    # Inicializar servicio de factores
    try:
        factor_service = FactorService(settings)
        app.state.factor_service = factor_service
        LOGGER.info("Servicio de factores inicializado")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Error inicializando servicio de factores: %s", exc)
        app.state.factor_service = None

    # Inicializar servicio de valuación
    try:
        valuation_service = ValuationService(settings)
        app.state.valuation_service = valuation_service
        LOGGER.info("Servicio de valuación inicializado")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Error inicializando servicio de valuación: %s", exc)
        app.state.valuation_service = None

    checkpoint_path = os.getenv("CARIA_MODEL_CHECKPOINT")
    model_bundle = None
    if checkpoint_path and load_model_bundle is not None:
        try:
            model_bundle = load_model_bundle(settings, checkpoint_path)
            LOGGER.info("Checkpoint de modelo cargado desde %s", checkpoint_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("No se pudo cargar el checkpoint del modelo: %s", exc)
    elif checkpoint_path and load_model_bundle is None:
        LOGGER.warning("Variable CARIA_MODEL_CHECKPOINT configurada pero modelo legacy no disponible (pytorch_lightning no instalado)")
    else:
        LOGGER.info("Variable CARIA_MODEL_CHECKPOINT no configurada; endpoints de predicción legacy desactivados")

    app.state.model_bundle = model_bundle


app = FastAPI(
    title="Caria API",
    description="Multi-user investment analysis platform with regime detection, RAG, and valuation",
    version="2.0.0"
)

# Configurar CORS
# Accept specific origins from env + all Vercel deployments via regex
cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
cors_origins = [origin.strip().lower() for origin in cors_origins_env.split(",") if origin.strip()]

# Allow all Vercel deployments (*.vercel.app) using regex
# This matches both preview and production deployments
vercel_regex = r"https://.*\.vercel\.app"

# Logging para debugging
LOGGER.info(f"CORS configured with origins: {cors_origins}")
LOGGER.info(f"CORS regex pattern for Vercel: {vercel_regex}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=vercel_regex,  # Accept all *.vercel.app domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

_init_state(app)

# Include domain routers per audit document (4.1): Modular monolith architecture
# Domain boundaries: Each domain is self-contained with strict boundaries
# Note: Individual routes already have /api/* prefixes, so domain routers include them as-is
app.include_router(identity_router)  # Identity domain: auth, users, sessions
app.include_router(portfolio_router)  # Portfolio domain: holdings, analytics, TAA, Monte Carlo
app.include_router(social_router)  # Social domain: community, chat
app.include_router(analysis_domain_router)  # Analysis domain: regime, factors, valuation, validation
app.include_router(market_data_router)  # Market Data domain: prices, indicators

# Legacy routes (kept for backward compatibility - will be deprecated)
# These routes are now organized under domain routers above
app.include_router(auth_router)  # Legacy: Use /api/auth instead
app.include_router(analysis_router)  # Legacy: Use /api/analysis instead
app.include_router(regime_router)  # Legacy: Use /api/regime instead
app.include_router(factors_router)  # Legacy: Use /api/factors instead
app.include_router(valuation_router)  # Legacy: Use /api/valuation instead
app.include_router(prices_router)  # Legacy: Use /api/prices instead
app.include_router(holdings_router)  # Legacy: Use /api/portfolio/holdings instead
app.include_router(chat_router)  # Legacy: Use /api/chat instead
app.include_router(community_router)  # Legacy: Use /api/community instead
app.include_router(portfolio_analytics_router)  # Legacy: Use /api/portfolio/analytics instead
app.include_router(monte_carlo_router)  # Legacy: Use /api/portfolio/monte-carlo instead
app.include_router(model_validation_router)  # Legacy: Use /api/validation instead
app.include_router(tactical_allocation_router)  # Legacy: Use /api/portfolio/tactical instead
app.include_router(ux_tracking_router)  # UX tracking: user journeys, onboarding metrics (per audit 4.2)

# Mount SocketIO app for WebSocket support per audit document
# This combines FastAPI with SocketIO to handle both HTTP and WebSocket connections
socketio_app = ASGIApp(sio, other_asgi_app=app)


# Helper function to check if origin is allowed (including regex patterns)
def is_origin_allowed(origin: str) -> bool:
    """Check if origin is allowed by CORS configuration."""
    if not origin:
        return False
    # Check exact match
    if origin.lower() in cors_origins:
        return True
    # Check regex pattern for Vercel
    if re.match(vercel_regex, origin):
        return True
    return False


# Exception handlers para asegurar que CORS headers se envíen incluso en errores
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with CORS headers."""
    response = JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
    # Agregar headers CORS manualmente si no están presentes
    origin = request.headers.get("origin")
    if origin and is_origin_allowed(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


def _make_serializable(obj):
    """Recursively convert objects to JSON-serializable format."""
    if isinstance(obj, Exception):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Fallback: convert to string
        return str(obj)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with CORS headers."""
    # Convert errors to serializable format recursively
    errors = exc.errors()
    if errors:
        # Extract error message safely
        first_error = errors[0]
        error_detail = first_error.get("msg", "Validation error")
        # Convert to string if it's an Exception
        if isinstance(error_detail, Exception):
            error_detail = str(error_detail)
        else:
            error_detail = str(error_detail)
    else:
        error_detail = "Validation error"
    
    # Ensure errors list is fully serializable (including nested ctx)
    serializable_errors = [_make_serializable(err) for err in errors]
    
    response = JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": error_detail, "errors": serializable_errors},
    )
    # Agregar headers CORS manualmente
    origin = request.headers.get("origin")
    if origin and is_origin_allowed(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with CORS headers."""
    LOGGER.exception("Unhandled exception: %s", exc)
    # Ensure error detail is serializable
    error_detail = str(exc) if exc else "Internal server error"
    response = JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": error_detail},
    )
    # Agregar headers CORS manualmente
    origin = request.headers.get("origin")
    if origin and is_origin_allowed(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


@app.get("/health/live")
def healthcheck_live() -> dict[str, str]:
    """Liveness probe: verifica que la aplicación responde."""
    return {"status": "ok"}


@app.get("/health/ready")
def healthcheck_ready() -> dict[str, str]:
    """Readiness probe: verifica que todas las dependencias están disponibles."""
    status_dict = healthcheck()
    # Verificar que servicios críticos están disponibles
    if status_dict.get("database") == "unavailable":
        raise HTTPException(status_code=503, detail="Database unavailable")
    return status_dict


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Healthcheck que muestra estado de todos los servicios."""
    status = {"status": "ok"}

    # Database / Authentication
    try:
        import psycopg2
        from urllib.parse import urlparse, parse_qs

        # Try DATABASE_URL first (Cloud SQL format)
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            parsed = urlparse(database_url)
            query_params = parse_qs(parsed.query)

            # Check for Unix socket (Cloud SQL)
            unix_socket_host = query_params.get('host', [None])[0]

            if unix_socket_host:
                # Use Cloud SQL Unix socket
                conn = psycopg2.connect(
                    host=unix_socket_host,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip('/'),
                )
            elif parsed.hostname:
                # Use normal TCP connection
                conn = psycopg2.connect(
                    host=parsed.hostname,
                    port=parsed.port or 5432,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path.lstrip('/'),
                )
            else:
                raise ValueError("Invalid DATABASE_URL format")
            conn.close()
            status["database"] = "available"
            status["auth"] = "available"
        elif os.getenv("POSTGRES_PASSWORD"):
            # Fallback to individual variables
            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                user=os.getenv("POSTGRES_USER", "caria_user"),
                password=os.getenv("POSTGRES_PASSWORD"),
                database=os.getenv("POSTGRES_DB", "caria"),
            )
            conn.close()
            status["database"] = "available"
            status["auth"] = "available"
        else:
            LOGGER.warning("Neither DATABASE_URL nor POSTGRES_PASSWORD set, database check skipped")
            status["database"] = "unconfigured"
            status["auth"] = "unconfigured"
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Database connection failed: %s", exc)
        status["database"] = "unavailable"
        status["auth"] = "unavailable"

    # RAG (Sistema II)
    if app.state.vector_store is None:
        status["rag"] = "disabled"
    else:
        status["rag"] = "available"

    # Régimen HMM (Sistema I)
    regime_service = getattr(app.state, "regime_service", None)
    if regime_service and regime_service.is_available():
        status["regime"] = "available"
    else:
        status["regime"] = "unavailable"

    # Factores (Sistema III)
    factor_service = getattr(app.state, "factor_service", None)
    if factor_service:
        status["factors"] = "available"
    else:
        status["factors"] = "unavailable"

    # Valuación (Sistema IV)
    valuation_service = getattr(app.state, "valuation_service", None)
    if valuation_service:
        status["valuation"] = "available"
    else:
        status["valuation"] = "unavailable"

    # Modelo legacy (deprecated)
    if app.state.model_bundle is None:
        status["legacy_model"] = "unavailable"
    else:
        status["legacy_model"] = "available"

    return status


# Export socketio_app as the main ASGI application for uvicorn
# This combines FastAPI (HTTP) with SocketIO (WebSocket) per audit document
# uvicorn will use: uvicorn api.app:socketio_app
__all__ = ['socketio_app', 'app']

