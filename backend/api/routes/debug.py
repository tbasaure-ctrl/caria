from fastapi import APIRouter, HTTPException
import logging
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
        import os
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return {"status": "error", "detail": "GEMINI_API_KEY is missing or empty"}
        
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        
        # 2. Attempt simple call
        prompt = "Hello, are you working?"
        LOGGER.info(f"Testing Gemini with key: {masked_key}")
        
        response = llm_service.call_llm(prompt)
        
        if response:
            return {
                "status": "success",
                "provider": "Gemini",
                "response": response,
                "api_key_masked": masked_key
            }
        else:
            # If None, it failed inside call_llm. We need to check logs or try to capture the error here.
            # Since call_llm catches exceptions, we might want to call _call_gemini directly to see the error?
            # Or we rely on the logs we just added.
            # Let's try to call _call_gemini directly to capture the exception if possible, 
            # but _call_gemini is internal.
            
            # Better: let's try a direct request here to show the user the error
            import requests
            api_url = os.getenv(
                "GEMINI_API_URL",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            )
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key.strip(),
            }
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            
            try:
                resp = requests.post(api_url, headers=headers, json=payload, timeout=10)
                return {
                    "status": "failed",
                    "http_status": resp.status_code,
                    "raw_response": resp.text,
                    "api_key_masked": masked_key
                }
            except Exception as e:
                return {
                    "status": "exception",
                    "detail": str(e),
                    "api_key_masked": masked_key
                }

    except Exception as e:
        LOGGER.exception("Debug LLM failed")
        raise HTTPException(status_code=500, detail=str(e))
