"""
LLM Service for handling RAG and interactions with Gemini/Llama.
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
        Call Gemini, fallback to Llama.
        Returns the raw text response.
        """
        # Try Gemini first
        response = self._call_gemini(prompt, system_prompt)
        if response:
            return response
            
        # Fallback to Llama
        return self._call_llama(prompt, system_prompt)

    def _call_gemini(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        api_url = os.getenv(
            "GEMINI_API_URL",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        )

        if not api_key:
            LOGGER.warning("GEMINI_API_KEY no configurada")
            return None

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        
        # Construct payload
        # Gemini API structure might vary, sticking to the one in analysis.py
        # System prompt integration depends on model version, but we'll append to prompt for simplicity if needed
        # or use the proper field if we were using the SDK. REST API supports system_instruction in some versions.
        # For safety, we'll prepend system prompt to user prompt if provided, or keep it simple.
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser Query:\n{prompt}"

        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}]
        }

        for attempt in range(3):
            try:
                resp = requests.post(api_url, headers=headers, json=payload, timeout=40)
                if resp.status_code == 503 and attempt < 2:
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                return text if text else None
            except Exception as e:
                LOGGER.warning(f"Gemini attempt {attempt+1} failed: {e}")
                break
        return None

    def _call_llama(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        api_key = os.getenv("LLAMA_API_KEY")
        api_url = os.getenv("LLAMA_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        model = os.getenv("LLAMA_MODEL", "llama-3.1-8b-instruct")

        if not api_key or not api_url:
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
