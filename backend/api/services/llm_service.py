"""
LLM Service for handling RAG interactions.
Supports Groq (Llama), Claude, and Gemini as fallbacks.
"""
import json
import logging
import os
from typing import Any, List, Optional, Tuple

import requests
import google.generativeai as genai

LOGGER = logging.getLogger("caria.api.services.llm")
DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

class LLMService:
    def __init__(self, retriever=None, embedder=None):
        self.retriever = retriever
        self.embedder = embedder
        
        # Initialize Gemini if key exists
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)

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
        Call LLM with fallback chain: Groq -> Claude -> Gemini.
        """
        # 1. Try Claude if structured output needed or fallback requested
        if use_fallback or self._needs_structured_output(prompt, system_prompt):
            result = self._call_claude(prompt, system_prompt)
            if result: return result
            LOGGER.warning("Claude fallback failed, trying Groq...")
        
        # 2. Try Groq (Llama)
        result = self._call_llama(prompt, system_prompt)
        if result: return result
        
        # 3. Try Claude (if not tried yet)
        if not use_fallback:
            LOGGER.info("Groq failed, trying Claude...")
            result = self._call_claude(prompt, system_prompt)
            if result: return result

        # 4. Try Gemini (Last resort / Free tier often available)
        LOGGER.info("All primary models failed, trying Gemini...")
        result = self._call_gemini(prompt, system_prompt)
        if result: return result
        
        return "Lo siento, el servicio de análisis no está disponible en este momento (API Keys missing or quota exceeded)."
    
    def _needs_structured_output(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
        combined = (system_prompt or "") + " " + prompt
        keywords = ["json", "structured", "format", "schema", "parse", "extract"]
        return any(keyword in combined.lower() for keyword in keywords)

    def _call_llama(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        api_key = os.getenv("LLAMA_API_KEY")
        api_url = os.getenv("LLAMA_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        model = os.getenv("LLAMA_MODEL", DEFAULT_GROQ_MODEL)

        if not api_key: return None

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._invoke_groq(api_url, headers, messages, model)

    def _invoke_groq(self, api_url: str, headers: dict, messages: list, model: str) -> Optional[str]:
        try:
            resp = requests.post(api_url, headers=headers, json={"model": model, "messages": messages}, timeout=40)
            resp.raise_for_status()
            return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            LOGGER.warning(f"Groq call failed: {e}")
            return None
    
    def _call_claude(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key: return None
        
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                    "system": system_prompt
                },
                timeout=60
            )
            resp.raise_for_status()
            content = resp.json().get("content", [])
            if content: return content[0].get("text", "")
        except Exception as e:
            LOGGER.warning(f"Claude call failed: {e}")
            return None
        return None

    def _call_gemini(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Fallback to Google Gemini."""
        if not self.gemini_key:
            LOGGER.warning("GEMINI_API_KEY not configured")
            return None
            
        try:
            model = genai.GenerativeModel('gemini-2.0-flash') # Use efficient model
            
            # Combine system prompt if present
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System Instruction: {system_prompt}\n\nUser Request: {prompt}"
                
            response = model.generate_content(full_prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            LOGGER.warning(f"Gemini call failed: {e}")
            return None
        return None

    def parse_json_response(self, text: str) -> dict:
        try:
            clean_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception:
            return {}
