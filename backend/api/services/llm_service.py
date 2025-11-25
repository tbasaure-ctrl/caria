"""
LLM Service for handling RAG interactions.
Uses OpenAI-compatible API (e.g. Groq) as primary provider.
"""
import json
import logging
import os
from typing import Any, List, Optional, Tuple

import requests

LOGGER = logging.getLogger("caria.api.services.llm")

class LLMService:
    def __init__(self, retriever=None, embedder=None):
        self.retriever = retriever
        self.embedder = embedder
        
        # Load configuration
        self.api_key = os.getenv("OPENAI_API_KEY")
        # Default to Groq if not set, but respect env var
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1/chat/completions") 
        self.model = os.getenv("OPENAI_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

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
            results = self.retriever.query(embedding, top_k=top_k)
            
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
        Call LLM using OpenAI-compatible endpoint.
        """
        if not self.api_key:
            return "Error: API Key no configurada."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # Normalize URL: strip trailing slashes, then append /chat/completions if missing
            url = self.base_url.rstrip("/")
            if "chat/completions" not in url:
                url += "/chat/completions"
            
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
