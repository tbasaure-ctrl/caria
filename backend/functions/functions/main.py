"""
Firebase Functions para Wise Adviser
Migración de endpoints que usan Gemini API a serverless
Con integración RAG llamando al backend tradicional
"""
import json
import os
import logging
from typing import Any

import requests
from firebase_functions import https_fn, options

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Leer configuración de Firebase Functions
# Firebase Functions v1 usa functions.config(), v2 usa variables de entorno
try:
    import functions_framework
    from firebase_functions import functions_config
    _config = functions_config.config()
except (ImportError, AttributeError):
    # Fallback a variables de entorno (Firebase Functions v2 o desarrollo local)
    _config = None


def _get_config(key: str, default: str = None) -> str:
    """
    Obtiene configuración de Firebase Functions config o variables de entorno
    """
    if _config:
        # Firebase Functions v1 - usar functions.config()
        # functions.config() devuelve un objeto con atributos anidados
        try:
            keys = key.split('.')
            value = _config
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                else:
                    value = None
                    break
            if value:
                return str(value)
        except Exception:
            pass
    
    # Fallback a variables de entorno
    env_key = key.upper().replace('.', '_')
    return os.environ.get(env_key, default)


def _get_rag_chunks(query: str, top_k: int, backend_url: str) -> list[dict[str, Any]]:
    """
    Obtiene chunks de RAG llamando al endpoint /api/analysis/wisdom del backend tradicional
    """
    try:
        # Llamar al endpoint de wisdom del backend
        response = requests.post(
            f"{backend_url}/api/analysis/wisdom",
            json={
                "query": query,
                "top_k": top_k,
                "page": 1,
                "page_size": top_k
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Extraer resultados
        results = data.get("results", [])
        return results
    except Exception as e:
        logger.warning(f"Error obteniendo chunks RAG: {e}")
        return []


def _call_gemini(prompt: str) -> dict[str, Any] | None:
    """
    Llama a Gemini API - misma lógica que tu código actual
    """
    api_key = _get_config("gemini.api_key") or os.environ.get("GEMINI_API_KEY")
    api_url = _get_config(
        "gemini.api_url",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    ) or os.environ.get(
        "GEMINI_API_URL",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    )
    
    if not api_key:
        logger.warning("GEMINI_API_KEY no configurada")
        return None

    try:
        resp = requests.post(
            api_url,
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=40,
        )
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return {"raw_text": text}
    except Exception as exc:
        logger.exception(f"Error llamando a Gemini: {exc}")
        return None


def _call_llama(prompt: str) -> dict[str, Any] | None:
    """
    Fallback a Llama - misma lógica que tu código actual
    """
    api_key = _get_config("llama.api_key") or os.environ.get("LLAMA_API_KEY")
    api_url = _get_config("llama.api_url") or os.environ.get("LLAMA_API_URL")
    model = _get_config("llama.model_name", "llama-3.1-70b-instruct") or os.environ.get("LLAMA_MODEL_NAME", "llama-3.1-70b-instruct")

    if not api_key or not api_url:
        logger.info("Llama no configurado")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an investment coach that analyses theses, finds biases and gives practical recommendations.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=40)
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return {"raw_text": text}
    except Exception as exc:
        logger.exception(f"Error llamando a Llama: {exc}")
        return None


def _parse_llm_json(raw_text: str) -> tuple[str, list[str], list[str], float]:
    """
    Parsea la respuesta del LLM esperando JSON
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
        return raw_text.strip(), [], [], 0.6


def _add_cors_headers(response: https_fn.Response, origin: str = None) -> https_fn.Response:
    """Agrega headers CORS a la respuesta"""
    allowed_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:80",
        # Agrega aquí tus dominios de producción
    ]
    
    if origin and origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"
    
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Max-Age"] = "3600"
    
    return response


@https_fn.on_request()
def challenge_thesis(req: https_fn.Request) -> https_fn.Response:
    """
    Endpoint serverless que reemplaza: POST /api/analysis/challenge
    
    Desafía una tesis de inversión usando RAG + Gemini/Llama
    """
    origin = req.headers.get("origin", "")
    
    if req.method == "OPTIONS":
        response = https_fn.Response("", status=204)
        return _add_cors_headers(response, origin)
    
    try:
        # Parsear request body
        data = req.get_json(silent=True)
        if not data:
            response = https_fn.Response(
                json.dumps({"error": "Request body requerido"}),
                status=400,
                headers={"Content-Type": "application/json"}
            )
            return _add_cors_headers(response, origin)
        
        thesis = data.get("thesis", "")
        ticker = data.get("ticker")
        top_k = data.get("top_k", 5)
        
        if not thesis or len(thesis) < 10:
            response = https_fn.Response(
                json.dumps({"error": "Tesis debe tener al menos 10 caracteres"}),
                status=400,
                headers={"Content-Type": "application/json"}
            )
            return _add_cors_headers(response, origin)
        
        # 1) Obtener chunks de RAG desde el backend tradicional
        backend_url = _get_config("backend.url", "http://localhost:8000") or os.environ.get("BACKEND_URL", "http://localhost:8000")
        query_text = f"{thesis}\nTicker: {ticker or ''}"
        
        retrieved_chunks = _get_rag_chunks(query_text, top_k, backend_url)
        
        # Construir texto de evidencia para el prompt
        evidence_lines: list[str] = []
        for chunk in retrieved_chunks[:5]:
            snippet = (chunk.get("content") or "").replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            source = chunk.get("source", "doc")
            title = chunk.get("title", "")
            evidence_lines.append(f"- [{source}] {title}: {snippet}")
        
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "No se encontraron fragmentos relevantes."
        
        # 2) Construir prompt con evidencia RAG
        ticker_str = ticker or "N/A"
        prompt = f"""
You are CARIA, a rational investment sparring partner.

User:
- Thesis: "{thesis}"
- Ticker: {ticker_str}

Evidence from historical wisdom, books, and notes (may be empty or noisy):
{evidence_text}

Tasks:
1. Critically analyze the thesis: quality of business, valuation, risks, time horizon, and portfolio sizing.
2. Explicitly identify cognitive biases (confirmation, optimism, anchoring, FOMO, etc.).
3. Give practical, concrete recommendations (what to double-check, what scenarios to run, when NOT to invest).
4. Estimate a confidence_score between 0 and 1 for how robust the thesis looks.

Respond ONLY in valid JSON with this exact structure:

{{
  "critical_analysis": "string, detailed but concise, in Spanish if the input is Spanish.",
  "identified_biases": ["string", "..."],
  "recommendations": ["string", "..."],
  "confidence_score": 0.0-1.0
}}
"""
        
        # 3) Llamar a Gemini con fallback a Llama
        llm_result = _call_gemini(prompt)
        if llm_result is None:
            llm_result = _call_llama(prompt)
        
        if llm_result is None or not llm_result.get("raw_text"):
            # Fallback básico
            response = https_fn.Response(
                json.dumps({
                    "thesis": thesis,
                    "retrieved_chunks": retrieved_chunks,
                    "critical_analysis": f"Análisis de la tesis sobre {ticker_str if ticker else 'N/A'}: \"{thesis}\". No se pudo acceder a un modelo de lenguaje avanzado.",
                    "identified_biases": [
                        "Confirmation bias: solo buscar información que confirma tu tesis",
                        "Overconfidence: subestimar riesgos y volatilidad",
                    ],
                    "recommendations": [
                        "Revisar múltiple evidencia independiente",
                        "Analizar estados financieros y valuación",
                        "Definir un rango de escenarios",
                        "Decidir un tamaño de posición acorde a tu convicción",
                    ],
                    "confidence_score": 0.5,
                }),
                status=200,
                headers={"Content-Type": "application/json"}
            )
            return _add_cors_headers(response, origin)
        
        # 4) Parsear respuesta del LLM
        critical, biases, recs, conf = _parse_llm_json(llm_result["raw_text"])
        
        response = https_fn.Response(
            json.dumps({
                "thesis": thesis,
                "retrieved_chunks": retrieved_chunks,
                "critical_analysis": critical,
                "identified_biases": biases,
                "recommendations": recs,
                "confidence_score": conf,
            }),
            status=200,
            headers={"Content-Type": "application/json"}
        )
        return _add_cors_headers(response, origin)
        
    except Exception as e:
        logger.exception(f"Error en challenge_thesis: {e}")
        response = https_fn.Response(
            json.dumps({"error": str(e)}),
            status=500,
            headers={"Content-Type": "application/json"}
        )
        return _add_cors_headers(response, origin)


@https_fn.on_request()
def analyze_with_gemini(req: https_fn.Request) -> https_fn.Response:
    """
    Endpoint simple para análisis directo con Gemini
    Reemplaza llamadas directas a Gemini desde el frontend
    """
    origin = req.headers.get("origin", "")
    
    if req.method == "OPTIONS":
        response = https_fn.Response("", status=204)
        return _add_cors_headers(response, origin)
    
    try:
        data = req.get_json(silent=True)
        if not data:
            response = https_fn.Response(
                json.dumps({"error": "Request body requerido"}),
                status=400,
                headers={"Content-Type": "application/json"}
            )
            return _add_cors_headers(response, origin)
        
        prompt = data.get("prompt", "")
        if not prompt:
            response = https_fn.Response(
                json.dumps({"error": "Prompt requerido"}),
                status=400,
                headers={"Content-Type": "application/json"}
            )
            return _add_cors_headers(response, origin)
        
        # Llamar a Gemini
        result = _call_gemini(prompt)
        if result is None:
            result = _call_llama(prompt)
        
        if result is None:
            response = https_fn.Response(
                json.dumps({"error": "No se pudo acceder a ningún modelo LLM"}),
                status=503,
                headers={"Content-Type": "application/json"}
            )
            return _add_cors_headers(response, origin)
        
        response = https_fn.Response(
            json.dumps({"raw_text": result.get("raw_text", "")}),
            status=200,
            headers={"Content-Type": "application/json"}
        )
        return _add_cors_headers(response, origin)
        
    except Exception as e:
        logger.exception(f"Error en analyze_with_gemini: {e}")
        response = https_fn.Response(
            json.dumps({"error": str(e)}),
            status=500,
            headers={"Content-Type": "application/json"}
        )
        return _add_cors_headers(response, origin)

