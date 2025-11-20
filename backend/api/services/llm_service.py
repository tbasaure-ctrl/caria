"""
LLM Service for handling RAG and interactions with Llama (Groq).
Gemini support removed due to Google project suspension.
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

    def get_rag_context(self, query: str, top_k: int = 5) -> Tuple[str, List[dict]]:
        """
        Retrieve relevant chunks from the knowledge base.
        Returns a tuple of (formatted_evidence_text, list_of_chunks).
        """
        if not self.retriever or not self.embedder:
            return "Stack RAG no disponible.", []

        try:
            embedding = self.embedder.embed_text(query)
            results = self.retriever.query(embedding, top_k=top_k)
            
            chunks = []
            evidence_lines = []
            
            for r in results:
                # Serialize result similar to analysis.py
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

    def call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        Call Llama (Groq) API only.
        Returns the raw text response.
        """
        # Use Llama only (Gemini removed due to Google project suspension)
        return self._call_llama(prompt, system_prompt)

    def _call_gemini(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        DEPRECATED: Gemini API support removed due to Google project suspension.
        This method is kept for backward compatibility but always returns None.
        Use _call_llama() instead.
        """
        LOGGER.warning("Gemini API is deprecated and disabled. Use Llama instead.")
        return None

    def _call_llama(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        Call Llama API via Groq (OpenAI-compatible endpoint).
        """
        api_key = os.getenv("LLAMA_API_KEY")
        api_url = os.getenv("LLAMA_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        model = os.getenv("LLAMA_MODEL", "llama-3.1-8b-instruct")

        if not api_key:
            LOGGER.warning("LLAMA_API_KEY not configured")
            return None

        if not api_url:
            LOGGER.warning("LLAMA_API_URL not configured")
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
        }

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=40)
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            LOGGER.warning(f"Llama call failed: {e}")
            return None

    def parse_json_response(self, text: str) -> dict:
        """Helper to parse JSON from LLM response."""
        try:
            # Clean markdown code blocks if present
            clean_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception:
            return {}
