"""Rutas para análisis de inversión (RAG + LLM)."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

# Configurar paths para encontrar caria
CURRENT_FILE = Path(__file__).resolve()
CARIA_DATA_SRC = CURRENT_FILE.parent.parent.parent.parent / "caria_data" / "src"
if CARIA_DATA_SRC.exists() and str(CARIA_DATA_SRC) not in sys.path:
    sys.path.insert(0, str(CARIA_DATA_SRC))

from api.dependencies import get_current_user
from api.validators import Ticker
from api.services.llm_service import LLMService
from caria.models.auth import UserInDB
from caria.retrieval.retrievers import RetrievalResult

LOGGER = logging.getLogger("caria.api.analysis")

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


# ---------- Modelos Pydantic que deben calzar con el frontend ----------


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., description="Vector de características alineado al modelo entrenado"
    )


class PredictionResponse(BaseModel):
    prediction: float


class WisdomQuery(BaseModel):
    query: str = Field(..., description="Consulta en lenguaje natural")
    top_k: int = Field(5, ge=1, le=20, description="Número de fragmentos a recuperar")
    page: int = Field(1, ge=1, description="Página actual (para paginación)")
    page_size: int = Field(5, ge=1, le=50, description="Resultados por página")


class WisdomResult(BaseModel):
    id: str
    score: float
    title: str | None = None
    source: str | None = None
    content: str | None = None
    metadata: dict[str, Any]


class WisdomResponse(BaseModel):
    results: list[WisdomResult]
    pagination: dict[str, int | bool]


class ChallengeThesisRequest(BaseModel):
    thesis: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Tesis de inversión a desafiar (10-2000 caracteres)",
    )
    ticker: Ticker | None = Field(
        None, description="Ticker opcional para enriquecer contexto (3-5 caracteres)"
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Número de fragmentos a recuperar (1-20)",
    )


class ChallengeThesisResponse(BaseModel):
    thesis: str
    retrieved_chunks: list[WisdomResult]
    critical_analysis: str
    identified_biases: list[str]
    recommendations: list[str]
    confidence_score: float


# ---------- Helpers de infraestructura ----------


def _guard_model_bundle(request: Request):
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Configura CARIA_MODEL_CHECKPOINT y reinicia el servicio.",
        )
    return bundle


def _serialize_result(result: RetrievalResult) -> WisdomResult:
    metadata = result.metadata.copy()
    return WisdomResult(
        id=result.id,
        score=float(result.score),
        title=metadata.get("title"),
        source=metadata.get("source"),
        content=metadata.get("content"),
        metadata=metadata,
    )


# ---------- Endpoints legacy de predicción / búsqueda ----------


@router.post("/predict", response_model=PredictionResponse)
def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    bundle = _guard_model_bundle(request)
    prediction = float(bundle.predict([payload.features])[0])
    return PredictionResponse(prediction=prediction)


@router.post("/wisdom", response_model=WisdomResponse)
def search_wisdom(request: Request, payload: WisdomQuery) -> WisdomResponse:
    """Search wisdom/knowledge base with simple pagination."""
    retriever = getattr(request.app.state, "retriever", None)
    embedder = getattr(request.app.state, "embedder", None)

    if retriever is None or embedder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stack RAG no disponible. Revisa la configuración de pgvector y embeddings.",
        )

    embedding = embedder.embed_text(payload.query)

    total_to_fetch = payload.top_k * payload.page_size
    all_results = retriever.query(embedding, top_k=total_to_fetch)

    total = len(all_results)
    start_idx = (payload.page - 1) * payload.page_size
    end_idx = start_idx + payload.page_size
    paginated_results = all_results[start_idx:end_idx]

    serialized = [_serialize_result(result) for result in paginated_results]
    return WisdomResponse(
        results=serialized,
        pagination={
            "total": total,
            "page": payload.page,
            "page_size": payload.page_size,
            "has_next": end_idx < total,
            "has_prev": payload.page > 1,
        },
    )


# ---------- LLM Helper (Groq Llama) ----------


def _call_llama(prompt: str, system_prompt: Optional[str] = None) -> dict[str, Any] | None:
    """Llama a Llama vía proveedor compatible con OpenAI (ej: Groq)."""
    api_key = os.getenv("LLAMA_API_KEY")
    api_url = os.getenv(
        "LLAMA_API_URL",
        "https://api.groq.com/openai/v1/chat/completions",
    )
    model = os.getenv("LLAMA_MODEL", "llama-3.1-8b-instruct")

    if not api_key:
        LOGGER.info("LLAMA_API_KEY no configurada; se omite Llama")
        return None

    if not api_url:
        LOGGER.info("LLAMA_API_URL no configurada; se omite Llama")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = [
        {
            "role": "system",
            "content": system_prompt
            or (
                "You are an investment coach that analyses theses, "
                "finds biases and gives practical recommendations."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    payload = {"model": model, "messages": messages, "temperature": 0.7}

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=40)
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not text:
            LOGGER.warning("Llama respondió sin texto útil: %s", data)
            return None
        return {"raw_text": text}
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Error llamando a Llama: %s", exc)
        return None


def _parse_llm_json(raw_text: str) -> tuple[str, list[str], list[str], float]:
    """
    Espera que el LLM devuelva un JSON con:
    {
      "critical_analysis": "...",
      "identified_biases": ["...", "..."],
      "recommendations": ["...", "..."],
      "confidence_score": 0.8
    }
    Si no es JSON válido, usa todo como critical_analysis.
    """
    try:
        data = json.loads(raw_text)
        critical = str(data.get("critical_analysis", "")).strip() or raw_text.strip()
        biases = [str(b).strip() for b in data.get("identified_biases", []) if str(b).strip()]
        recs = [str(r).strip() for r in data.get("recommendations", []) if str(r).strip()]
        conf = float(data.get("confidence_score", 0.6))
        conf = max(0.0, min(conf, 1.0))
        return critical, biases, recs, conf
    except Exception:
        # Si no es JSON, tratamos todo como análisis en texto libre.
        return raw_text.strip(), [], [], 0.6


# ---------- Endpoint principal: challenge_thesis ----------


@router.post("/challenge", response_model=ChallengeThesisResponse)
def challenge_thesis(
    request: Request,
    payload: ChallengeThesisRequest,
    current_user: UserInDB = Depends(get_current_user),
) -> ChallengeThesisResponse:
    """
    Desafía una tesis de inversión:
    - Usa RAG si está disponible para extraer sabiduría histórica / citas.
    - Llama a Llama (Groq). Si falla, usa un checklist básico.
    """
    # 1) RAG: recuperar fragmentos relevantes si el stack está disponible
    llm_service: Optional[LLMService] = getattr(request.app.state, "llm_service", None)
    retriever = getattr(request.app.state, "retriever", None)
    embedder = getattr(request.app.state, "embedder", None)

    retrieved_chunks: list[WisdomResult] = []
    evidence_text = "Stack RAG no disponible en este momento."
    query_text = f"{payload.thesis}\nTicker: {payload.ticker or ''}"

    if llm_service is not None:
        try:
            rag_text, chunk_dicts = llm_service.get_rag_context(query_text, top_k=payload.top_k)
            evidence_text = rag_text
            retrieved_chunks = [
                WisdomResult(
                    id=chunk.get("id") or str(idx),
                    score=float(chunk.get("score", 0.0)),
                    title=chunk.get("title"),
                    source=chunk.get("source"),
                    content=chunk.get("content"),
                    metadata=chunk.get("metadata") or {},
                )
                for idx, chunk in enumerate(chunk_dicts)
            ]
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Error durante recuperación RAG con LLM Service: %s", exc)
            evidence_text = "Error al recuperar evidencia; continúa sin RAG."
    elif retriever is not None and embedder is not None:
        try:
            embedding = embedder.embed_text(query_text)
            results = retriever.query(embedding, top_k=payload.top_k)

            for r in results:
                wr = _serialize_result(r)
                retrieved_chunks.append(wr)

            evidence_lines: list[str] = []
            for wr in retrieved_chunks[:5]:
                snippet = (wr.content or "").replace("\n", " ")
                if len(snippet) > 400:
                    snippet = snippet[:400] + "..."
                evidence_lines.append(
                    f"- [{wr.source or 'doc'}] {wr.title or ''}: {snippet}"
                )
            evidence_text = "\n".join(evidence_lines) or "No se encontraron fragmentos relevantes."
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Error durante recuperación RAG: %s", exc)
            evidence_text = "Error al recuperar evidencia; continúa sin RAG."

    # 2) Construir prompt para el LLM
    ticker_str = payload.ticker or "N/A"
    analysis_system_prompt = (
        "You are Caria, a legendary and wise investment mentor. "
        "You blend rigorous fundamental analysis with decades of portfolio management experience. "
        "Challenge investors with respect, highlight asymmetric risks, and always explain the why. "
        "Respond in the user's language (Spanish if the thesis is in Spanish)."
    )
    prompt = f"""
Contexto recuperado (puede estar vac�o o tener ruido):
{evidence_text}

Tesis del usuario:
- Thesis: "{payload.thesis}"
- Ticker: {ticker_str}

Instrucciones:
1. Analiza la calidad del negocio, valuaci�n, catalizadores, riesgos y sizing recomendado.
2. Se�ala sesgos cognitivos concretos (confirmation, optimism, anchoring, FOMO, etc.).
3. Entrega recomendaciones accionables (datos a validar, m�tricas a monitorear, escenarios).
4. Asigna un confidence_score entre 0 y 1 sobre la robustez de la tesis.

Responde SOLO en JSON v�lido con la forma:
{{
  "critical_analysis": "texto detallado",
  "identified_biases": ["bias 1", "bias 2"],
  "recommendations": ["reco 1", "reco 2"],
  "confidence_score": 0.0-1.0
}}
"""

    llm_response_text: Optional[str] = None
    llm_error: Optional[str] = None

    if llm_service is not None:
        llm_response_text = llm_service.call_llm(prompt, system_prompt=analysis_system_prompt)
        if not llm_response_text:
            llm_error = "LLM service returned empty response"
    else:
        llm_result = _call_llama(prompt, system_prompt=analysis_system_prompt)
        if llm_result is not None and llm_result.get("raw_text"):
            llm_response_text = llm_result["raw_text"]
        else:
            llm_error = "Groq Llama fallback returned no response"

    if llm_response_text:
        critical, biases, recs, conf = _parse_llm_json(llm_response_text)
        return ChallengeThesisResponse(
            thesis=payload.thesis,
            retrieved_chunks=retrieved_chunks,
            critical_analysis=critical,
            identified_biases=biases,
            recommendations=recs,
            confidence_score=conf,
        )

    ticker_info = f" sobre {payload.ticker}" if payload.ticker else ""
    basic_analysis = f"""An�lisis de la tesis{ticker_info}: "{payload.thesis}".

No se pudo acceder al modelo Groq Llama ({llm_error or 'motivo desconocido'}).
Se muestra un checklist b�sico para que revises tu propia tesis.
"""

    return ChallengeThesisResponse(
        thesis=payload.thesis,
        retrieved_chunks=retrieved_chunks,
        critical_analysis=basic_analysis,
        identified_biases=[
            "Confirmation bias: solo buscar informaci�n que confirma tu tesis",
            "Overconfidence: subestimar riesgos y volatilidad",
        ],
        recommendations=[
            "Revisar m�ltiple evidencia independiente (fuera de redes sociales)",
            "Analizar estados financieros y valuaci�n frente a su historia y pares",
            "Definir un rango de escenarios (pesimista, base, optimista) y su impacto",
            "Decidir un tama�o de posici�n acorde a tu convicci�n y a tu portafolio total",
        ],
        confidence_score=0.5,
    )
