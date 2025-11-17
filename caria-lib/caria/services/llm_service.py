"""Servicio LLM multi-provider (Llama, Gemini, OpenAI).

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
    LLAMA = "llama"  # Via Ollama local
    GEMINI = "gemini"  # Google Gemini API
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
        LLMProvider.LLAMA: ["llama3.2", "llama3.1", "llama3", "llama2"],
        LLMProvider.GEMINI: ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
        LLMProvider.OPENAI: ["gpt-4", "gpt-3.5-turbo"],
    }

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.LLAMA,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Inicializa el servicio LLM.

        Args:
            provider: Provider a usar (llama, gemini, openai)
            model: Modelo específico (si None, usa el primero disponible)
            api_key: API key (solo para Gemini/OpenAI)
        """
        self.provider = provider
        self.model = model or self.MODELS[provider][0]
        self.api_key = api_key or self._get_api_key_from_env()
        self._client = None

    def _get_api_key_from_env(self) -> str | None:
        """Obtiene API key desde variables de entorno."""
        if self.provider == LLMProvider.GEMINI:
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif self.provider == LLMProvider.OPENAI:
            return os.getenv("OPENAI_API_KEY")
        return None

    def _init_client(self) -> Any:
        """Inicializa cliente del provider."""
        if self._client is not None:
            return self._client

        if self.provider == LLMProvider.LLAMA:
            try:
                import ollama
                self._client = ollama
                LOGGER.info("Ollama client initialized for Llama")
            except ImportError:
                raise RuntimeError(
                    "Ollama not installed. Install with: pip install ollama\n"
                    "Also ensure Ollama is running locally: https://ollama.ai/"
                )

        elif self.provider == LLMProvider.GEMINI:
            try:
                import google.generativeai as genai
                if not self.api_key:
                    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
                genai.configure(api_key=self.api_key)
                self._client = genai
                LOGGER.info("Gemini client initialized")
            except ImportError:
                raise RuntimeError(
                    "Google Generative AI not installed. Install with: pip install google-generativeai"
                )

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
            elif self.provider == LLMProvider.GEMINI:
                return self._generate_gemini(client, prompt, max_tokens, temperature, system_prompt)
            elif self.provider == LLMProvider.OPENAI:
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
        """Genera con Llama via Ollama."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        )

        content = response.get("message", {}).get("content", "")
        tokens = response.get("eval_count", 0)

        return LLMResponse(
            content=content,
            provider=self.provider,
            model=self.model,
            tokens_used=tokens,
        )

    def _generate_gemini(
        self,
        client: Any,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str | None,
    ) -> LLMResponse:
        """Genera con Google Gemini."""
        model = client.GenerativeModel(self.model)

        # Gemini maneja system prompt como parte del contenido
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        response = model.generate_content(
            full_prompt,
            generation_config=generation_config,
        )

        content = response.text
        tokens = response.usage_metadata.total_token_count if hasattr(response, "usage_metadata") else None

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

        Prioridad: Llama (local) > Gemini (API) > OpenAI (API)
        """
        # 1. Intentar Llama (gratis, local)
        try:
            import ollama
            ollama.list()  # Test connection
            LOGGER.info("Auto-detected: Llama via Ollama")
            return cls(provider=LLMProvider.LLAMA)
        except Exception:
            pass

        # 2. Intentar Gemini (API key en env)
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            try:
                LOGGER.info("Auto-detected: Gemini (API key found)")
                return cls(provider=LLMProvider.GEMINI)
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
            "1. Install Ollama and run locally: https://ollama.ai/\n"
            "2. Set GEMINI_API_KEY environment variable\n"
            "3. Set OPENAI_API_KEY environment variable"
        )
