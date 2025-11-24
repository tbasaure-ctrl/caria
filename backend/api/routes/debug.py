import logging
import os

import requests
from fastapi import APIRouter, HTTPException

from api.dependencies import open_db_connection
from api.services.llm_service import LLMService

router = APIRouter(prefix="/api/debug", tags=["debug"])
LOGGER = logging.getLogger("caria.api.routes.debug")


@router.get("/llm")
async def debug_llm():
    """
    Debug endpoint to test LLM connectivity and return raw error details.
    """
    try:
        llm_service = LLMService()

        # 1. Check API Key presence
        api_key = os.getenv("LLAMA_API_KEY", "").strip()
        if not api_key:
            return {"status": "error", "detail": "LLAMA_API_KEY is missing or empty"}

        api_url = os.getenv(
            "LLAMA_API_URL",
            "https://api.groq.com/openai/v1/chat/completions",
        )
        model = os.getenv("LLAMA_MODEL", "llama-3.1-8b-instruct")
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"

        # 2. Attempt simple call through the shared service
        prompt = "Hello Groq Llama, are you online?"
        LOGGER.info("Testing Groq Llama with key %s", masked_key)

        response = llm_service.call_llm(prompt)

        if response:
            return {
                "status": "success",
                "provider": "Groq Llama",
                "response": response,
                "api_key_masked": masked_key,
            }

        # 3. Direct request for detailed error output
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Connectivity diagnostic."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=15)
            return {
                "status": "failed",
                "http_status": resp.status_code,
                "raw_response": resp.text,
                "api_key_masked": masked_key,
            }
        except Exception as exc:  # pragma: no cover - network failure path
            return {
                "status": "exception",
                "detail": str(exc),
                "api_key_masked": masked_key,
            }

    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Debug LLM failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/db/ping")
async def debug_db_ping():
    """Quickly validate database connectivity and surface underlying errors."""
    try:
        conn = open_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                value = cursor.fetchone()[0]
        finally:
            conn.close()
        return {"status": "ok", "result": value}
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("DB ping failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Database ping failed: {exc.__class__.__name__}: {exc}",
        ) from exc
