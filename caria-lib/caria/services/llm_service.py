"""Servicio LLM multi-provider (Llama/Groq, OpenAI).

Abstracción para usar diferentes LLMs de manera intercambiable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

LOGGER = logging.getLogger("caria.services.llm")


class LLMProvider(str, Enum):
    """Providers de LLM soportados."""
    LLAMA = "llama"  # Groq (OpenAI-compatible) o Ollama local
    OPENAI = "openai"  # OpenAI API (fallback)


@dataclass
class LLMResponse:
    """Respuesta de un LLM."""
    content: str
    provider: LLMProvider
    model: str
    tokens_used: int | None = None


class LLMService:
    """Servicio unificado para múltiples LLMs."""

    # Modelos disponibles por provider
    MODELS = {
        LLMProvider.LLAMA: ["llama-3.1-8b-instruct", "llama3.2", "llama3.1"],
        LLMProvider.OPENAI: ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
    }

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.LLAMA,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Inicializa el servicio LLM.

        Args:
            provider: Provider a usar (llama u openai)
            model: Modelo específico (si None, usa el primero disponible)
            api_key: API key (solo para Groq/OpenAI)
        """
        self.provider = provider
        self.model = model or self.MODELS[provider][0]
        self.api_key = api_key or self._get_api_key_from_env()
        self._client = None

    def _get_api_key_from_env(self) -> str | None:
        """Obtiene API key desde variables de entorno."""
        if self.provider == LLMProvider.OPENAI:
            return os.getenv("OPENAI_API_KEY")
        return os.getenv("LLAMA_API_KEY")

    def _init_client(self) -> Any:
        """Inicializa cliente del provider."""
        if self._client is not None:
            return self._client

        if self.provider == LLMProvider.LLAMA:
            # Prefer Groq HTTP API (OpenAI compatible). Fall back to Ollama if API key missing.
            if self.api_key:
                import requests  # lazy import

                self._client = {
                    "type": "groq",
                    "session": requests.Session(),
                    "url": os.getenv(
                        "LLAMA_API_URL",
                        "https://api.groq.com/openai/v1/chat/completions",
                    ),
                }
                LOGGER.info("Groq client initialized for Llama")
            else:
                try:
                    import ollama

                    self._client = {"type": "ollama", "client": ollama}
                    LOGGER.info("Ollama client initialized for Llama (no API key set)")
                except ImportError as exc:
                    raise RuntimeError(
                        "Neither Groq LLAMA_API_KEY nor Ollama are available. "
                        "Set LLAMA_API_KEY for Groq or install Ollama locally."
                    ) from exc

        elif self.provider == LLMProvider.OPENAI:
            try:
                from openai import OpenAI
                if not self.api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                self._client = OpenAI(api_key=self.api_key)
                LOGGER.info("OpenAI client initialized")
            except ImportError:
                raise RuntimeError(
                    "OpenAI not installed. Install with: pip install openai"
                )

        return self._client

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Genera respuesta usando el LLM configurado.

        Args:
            prompt: Prompt del usuario
            max_tokens: Tokens máximos a generar
            temperature: Temperatura (0-1, mayor = más creativo)
            system_prompt: Instrucciones de sistema (opcional)

        Returns:
            LLMResponse con el contenido generado
        """
        client = self._init_client()

        try:
            if self.provider == LLMProvider.LLAMA:
                return self._generate_llama(client, prompt, max_tokens, temperature, system_prompt)
            if self.provider == LLMProvider.OPENAI:
                return self._generate_openai(client, prompt, max_tokens, temperature, system_prompt)
        except Exception as e:
            LOGGER.error("Error generating with %s: %s", self.provider, e)
            raise

    def _generate_llama(
        self,
        client: Any,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str | None,
    ) -> LLMResponse:
        """Genera con Llama via Groq (si hay API key) o Ollama."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if client["type"] == "groq":
            session = client["session"]
            resp = session.post(
                client["url"],
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=40,
            )
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            tokens = data.get("usage", {}).get("total_tokens")
        else:
            ollama_client = client["client"]
            response = ollama_client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            content = response.get("message", {}).get("content", "")
            tokens = response.get("eval_count", 0)

        return LLMResponse(
            content=content,
            provider=self.provider,
            model=self.model,
            tokens_used=tokens,
        )

    def _generate_openai(
        self,
        client: Any,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str | None,
    ) -> LLMResponse:
        """Genera con OpenAI."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = response.choices[0].message.content
        tokens = response.usage.total_tokens

        return LLMResponse(
            content=content,
            provider=self.provider,
            model=self.model,
            tokens_used=tokens,
        )

    @classmethod
    def auto_detect(cls) -> LLMService:
        """Detecta y usa el primer LLM disponible.

        Prioridad: Groq/Ollama > OpenAI.
        """
        # 1. Intentar Groq (LLAMA_API_KEY)
        if os.getenv("LLAMA_API_KEY"):
            LOGGER.info("Auto-detected: Groq (LLAMA_API_KEY found)")
            return cls(provider=LLMProvider.LLAMA)

        # 2. Intentar Ollama
        try:
            import ollama

            ollama.list()  # Test connection
            LOGGER.info("Auto-detected: Llama via Ollama")
            return cls(provider=LLMProvider.LLAMA)
        except Exception:
            pass

        # 3. Intentar OpenAI (fallback)
        if os.getenv("OPENAI_API_KEY"):
            try:
                LOGGER.info("Auto-detected: OpenAI (API key found)")
                return cls(provider=LLMProvider.OPENAI)
            except Exception:
                pass

        raise RuntimeError(
            "No LLM provider available. Options:\n"
            "1. Set LLAMA_API_KEY for Groq (https://console.groq.com/)\n"
            "2. Install Ollama and run locally: https://ollama.ai/\n"
            "3. Set OPENAI_API_KEY environment variable"
        )
