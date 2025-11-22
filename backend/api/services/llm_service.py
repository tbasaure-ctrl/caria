"""
LLM Service for handling RAG interactions with Groq's Llama endpoint.
Supports Llama 3.1 70B (default) via Groq with Claude 3.5 Sonnet as fallback.
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

    def call_llm(self, prompt: str, system_prompt: Optional[str] = None, use_fallback: bool = False) -> Optional[str]:
        """
        Call Groq's Llama endpoint (default) with Claude 3.5 Sonnet fallback.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            use_fallback: If True, skip Groq and use Claude directly (for structured JSON)
        
        Returns:
            LLM response text or None if both fail
        """
        # For structured JSON prompts, prefer Claude fallback
        if use_fallback or self._needs_structured_output(prompt, system_prompt):
            result = self._call_claude(prompt, system_prompt)
            if result:
                return result
            LOGGER.warning("Claude fallback failed, trying Groq...")
        
        # Try Groq first (default)
        result = self._call_llama(prompt, system_prompt)
        if result:
            return result
        
        # If Groq fails and we haven't tried Claude yet, use Claude as fallback
        if not use_fallback:
            LOGGER.info("Groq failed, falling back to Claude...")
            return self._call_claude(prompt, system_prompt)
        
        return None
    
    def _needs_structured_output(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
        """Detect if prompt requires structured JSON output."""
        combined = (system_prompt or "") + " " + prompt
        keywords = ["json", "structured", "format", "schema", "parse", "extract"]
        return any(keyword in combined.lower() for keyword in keywords)

    def _call_llama(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        Call Llama 3.1 70B via Groq (OpenAI-compatible endpoint).
        """
        api_key = os.getenv("LLAMA_API_KEY")
        api_url = os.getenv("LLAMA_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        # Default to Llama 3.1 70B for better quality
        # Groq model names: llama-3.1-70b-versatile, llama-3.1-70b-instruct, llama-3-70b-8192
        model = os.getenv("LLAMA_MODEL", "llama-3.1-70b-versatile")

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
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            LOGGER.info(f"Groq Llama 3.1 70B response received (model: {model})")
            return content
        except Exception as e:
            LOGGER.warning(f"Llama call failed: {e}")
            return None
    
    def _call_claude(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        Call Claude 3.5 Sonnet via Anthropic API (fallback for structured JSON).
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            LOGGER.warning("ANTHROPIC_API_KEY not configured, Claude fallback unavailable")
            return None
        
        api_url = "https://api.anthropic.com/v1/messages"
        model = "claude-3-5-sonnet-20241022"
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        
        # Build messages - Anthropic uses different format
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": model,
            "max_tokens": 4096,
            "messages": messages,
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            
            # Anthropic returns content as array of text blocks
            content_blocks = data.get("content", [])
            if content_blocks and len(content_blocks) > 0:
                content = content_blocks[0].get("text", "")
                LOGGER.info(f"Claude 3.5 Sonnet response received")
                return content
            return None
        except Exception as e:
            LOGGER.warning(f"Claude call failed: {e}")
            return None

    def parse_json_response(self, text: str) -> dict:
        """Helper to parse JSON from LLM response."""
        try:
            # Clean markdown code blocks if present
            clean_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception:
            return {}
    
    def call_llm_with_json_fallback(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        Call LLM with automatic fallback to Claude for structured JSON prompts.
        This is optimized for prompts that require JSON output.
        """
        # For JSON prompts, prefer Claude first (better structured output)
        if self._needs_structured_output(prompt, system_prompt):
            result = self._call_claude(prompt, system_prompt)
            if result:
                return result
            LOGGER.warning("Claude failed for JSON prompt, trying Groq...")
        
        # Fallback to Groq
        return self._call_llama(prompt, system_prompt)
