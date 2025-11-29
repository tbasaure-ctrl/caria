"""
LLM Service for handling RAG interactions.
Uses OpenAI-compatible API (e.g. Groq) as primary provider.
"""
import json
import logging
import os
from typing import Any, List, Optional, Tuple

import requests

try:
    from caria.services.prompt_builder_service import PromptBuilderService
    PROMPT_BUILDER_AVAILABLE = True
except ImportError:
    PROMPT_BUILDER_AVAILABLE = False

LOGGER = logging.getLogger("caria.api.services.llm")

class LLMService:
    def __init__(self, retriever=None, embedder=None):
        self.retriever = retriever
        self.embedder = embedder
        
        # Load configuration
        self.api_key = os.getenv("OPENAI_API_KEY")
        # xAI Grok 4 API endpoint (OpenAI-compatible)
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.x.ai/v1/chat/completions") 
        self.model = os.getenv("OPENAI_MODEL", "grok-4-0709")  # xAI Grok 4
        
        # Initialize PromptBuilder for liquidity-aware prompts
        if PROMPT_BUILDER_AVAILABLE:
            self.prompt_builder = PromptBuilderService()
            LOGGER.info("PromptBuilderService initialized - Hydraulic Score will be injected into prompts")
        else:
            self.prompt_builder = None
            LOGGER.warning("PromptBuilderService not available - prompts will not be liquidity-aware")

        if not self.api_key:
            LOGGER.warning("OPENAI_API_KEY not set. LLM calls will fail.")

    def get_rag_context(self, query: str, top_k: int = 5) -> Tuple[str, List[dict]]:
        """
        Retrieve relevant chunks from the knowledge base.
        """
        if not self.retriever or not self.embedder:
            return "Stack RAG no disponible.", []

        try:
            embedding = self.embedder.embed_text(query)
            # Retry logic for PostgreSQL connection issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    results = self.retriever.query(embedding, top_k=top_k)
                    break
                except Exception as db_error:
                    error_str = str(db_error)
                    error_type = type(db_error).__name__
                    is_ssl_error = (
                        "SSL connection" in error_str or 
                        "OperationalError" in error_type or
                        "connection" in error_str.lower() and "closed" in error_str.lower()
                    )
                    if is_ssl_error:
                        if attempt < max_retries - 1:
                            import time
                            wait_time = (attempt + 1) * 0.5  # Exponential backoff: 0.5s, 1s, 1.5s
                            LOGGER.warning(
                                f"Database connection error (attempt {attempt + 1}/{max_retries}), "
                                f"retrying in {wait_time}s... Error: {error_str[:100]}"
                            )
                            time.sleep(wait_time)
                            continue
                        else:
                            LOGGER.error(f"Database connection failed after {max_retries} attempts: {db_error}")
                            return "Error de conexión a la base de datos. Intenta nuevamente.", []
                    else:
                        raise
            
            chunks = []
            evidence_lines = []
            
            for r in results:
                metadata = r.metadata.copy()
                chunk = {
                    "id": r.id,
                    "score": float(r.score),
                    "title": metadata.get("title"),
                    "source": metadata.get("source"),
                    "content": metadata.get("content"),
                    "metadata": metadata
                }
                chunks.append(chunk)
                
                snippet = (chunk["content"] or "").replace("\n", " ")
                if len(snippet) > 400:
                    snippet = snippet[:400] + "..."
                evidence_lines.append(
                    f"- [{chunk['source'] or 'doc'}] {chunk['title'] or ''}: {snippet}"
                )
            
            evidence_text = "\n".join(evidence_lines) or "No se encontraron fragmentos relevantes."
            return evidence_text, chunks

        except Exception as e:
            LOGGER.exception(f"Error retrieving RAG context: {e}")
            return "Error al recuperar evidencia.", []

    def call_llm(self, prompt: str, system_prompt: Optional[str] = None, use_fallback: bool = False) -> Optional[str]:
        """
        Call LLM using OpenAI-compatible endpoint (Grok 4).
        Automatically injects Hydraulic Score context into system prompts.
        """
        if not self.api_key:
            return "Error: API Key no configurada."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # HYDRAULIC INJECTION: Enhance system prompt with liquidity context
        if self.prompt_builder and system_prompt:
            try:
                system_prompt = self.prompt_builder.build_context_aware_prompt(system_prompt, prompt)
                LOGGER.info("Injected Hydraulic Score into system prompt")
            except Exception as e:
                LOGGER.error(f"Failed to inject liquidity context: {e}")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # Handle full URL vs base URL
            # If base_url ends with /chat/completions, use it directly.
            # If it looks like a base (e.g. .../v1), append /chat/completions if missing?
            # User provided: https://api.groq.com/openai/v1/chat/completions
            # We'll use it as is if it looks complete.
            
            url = self.base_url
            if "chat/completions" not in url:
                 url = url.rstrip("/") + "/chat/completions"
            
            resp = requests.post(
                url, 
                headers=headers, 
                json={
                    "model": self.model, 
                    "messages": messages,
                    "temperature": 0.3
                }, 
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
        except Exception as e:
            LOGGER.error(f"LLM call failed: {e}")
            if hasattr(e, 'response') and e.response:
                LOGGER.error(f"Response: {e.response.text}")
            return "Lo siento, hubo un error al procesar tu solicitud con el modelo."

    def parse_json_response(self, text: str) -> dict:
        try:
            clean_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception:
            return {}
