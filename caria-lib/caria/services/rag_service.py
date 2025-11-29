# caria/services/rag_service.py

from __future__ import annotations

import os
import textwrap
import json
from dataclasses import dataclass
from typing import Any, List, Dict

import requests

from caria.retrieval.vector_store import VectorStore
from caria.retrieval.retrievers import Retriever
from caria.embeddings.generator import EmbeddingGenerator
from caria.config.settings import Settings


@dataclass
class ChallengeThesisResult:
    thesis: str
    retrieved_chunks: List[Dict[str, Any]]
    critical_analysis: str
    identified_biases: List[str]
    recommendations: List[str]
    confidence_score: float


class RAGService:
    """
    Servicio RAG para desafiar tesis de inversión usando Grok 4 (xAI)
    con API OpenAI-compatible. Si no está disponible, aplica un checklist básico.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        retriever: Retriever,
        embedder: EmbeddingGenerator,
        settings: Settings,
    ) -> None:
        self.vector_store = vector_store
        self.retriever = retriever
        self.embedder = embedder
        self.settings = settings

        # ---------- xAI Grok 4 (OpenAI Compatible) ----------
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.x.ai/v1/chat/completions")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "grok-4-0709")
        
        self.llm_available = bool(self.api_key)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _retrieve_chunks(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        embedding = self.embedder.embed_text(query)
        results = self.retriever.query(embedding, top_k=top_k)

        chunks: List[Dict[str, Any]] = []
        for r in results:
            meta = r.metadata or {}
            chunks.append(
                {
                    "id": r.id,
                    "score": float(r.score),
                    "title": meta.get("title"),
                    "source": meta.get("source"),
                    "content": meta.get("content"),
                    "metadata": meta,
                }
            )
        return chunks

    def _basic_fallback(
        self, thesis: str, ticker: str | None, chunks: List[Dict[str, Any]]
    ) -> ChallengeThesisResult:
        ticker_info = f" sobre {ticker}" if ticker else ""
        critical = textwrap.dedent(
            f"""
            Análisis de la tesis{ticker_info}: '{thesis}'.
            Se encontraron {len(chunks)} fragmentos relevantes de sabiduría histórica.
            Sin acceso a un modelo de lenguaje avanzado, este análisis es solo un checklist básico.
            """
        ).strip()

        biases = [
            "Confirmation bias: Verificar si solo buscas información que confirma tu tesis",
            "Overconfidence: Considerar si estás subestimando los riesgos y la volatilidad",
        ]
        recs = [
            "Revisar múltiples perspectivas antes de invertir",
            "Considerar el contexto macroeconómico actual",
            "Evaluar la valuación relativa vs. histórico y pares",
        ]

        return ChallengeThesisResult(
            thesis=thesis,
            retrieved_chunks=chunks,
            critical_analysis=critical,
            identified_biases=biases,
            recommendations=recs,
            confidence_score=0.5,
        )

    def _build_prompt(self, thesis: str, ticker: str | None, chunks: List[Dict[str, Any]]) -> str:
        context_blocks = []
        for i, ch in enumerate(chunks, start=1):
            if not ch.get("content"):
                continue
            context_blocks.append(
                f"[{i}] {ch.get('title') or 'Untitled'} – {ch['content']}"
            )
        context_text = "\n\n".join(context_blocks) if context_blocks else "N/A"

        system_instructions = textwrap.dedent(
            """
            Eres un analista de inversiones racional, tipo Buffett/Munger,
            cuya tarea es DESAFIAR una tesis de inversión del usuario.

            Debes:
            - Evaluar la calidad del negocio, la valuación y el momentum con el contexto histórico dado.
            - Buscar activamente riesgos, sesgos y puntos ciegos.
            - Hablar de forma clara, directa y educada, en el mismo idioma que la tesis (si está en inglés, respondes en inglés; si está en español, respondes en español).

            Devuelve SIEMPRE una respuesta en JSON válido con esta forma exacta:

            {
              "critical_analysis": "<texto en varios párrafos>",
              "identified_biases": ["bias 1", "bias 2", ...],
              "recommendations": ["recomendación 1", "recomendación 2", ...],
              "confidence_score": 0.0-1.0
            }

            No incluyas comentarios fuera del JSON.
            """
        ).strip()

        user_prompt = textwrap.dedent(
            f"""
            Tesis del usuario:
            \"\"\"{thesis}\"\"\"

            Ticker (si aplica): {ticker or "N/A"}

            Fragmentos de contexto (sabiduría histórica, libros de inversión, citas, etc.):
            {context_text}
            """
        ).strip()

        return system_instructions + "\n\n" + user_prompt

    def _parse_json_from_text(self, text: str) -> Dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("LLM response did not contain JSON.")
        json_str = text[start : end + 1]
        return json.loads(json_str)

    def _normalize_output(
        self,
        thesis: str,
        ticker: str | None,
        chunks: List[Dict[str, Any]],
        data: Dict[str, Any],
    ) -> ChallengeThesisResult:
        critical = str(data.get("critical_analysis", "")).strip()
        biases = data.get("identified_biases") or []
        recs = data.get("recommendations") or []
        conf = float(data.get("confidence_score", 0.5))

        biases = [str(b).strip() for b in biases if str(b).strip()]
        recs = [str(r).strip() for r in recs if str(r).strip()]
        if not (0.0 <= conf <= 1.0):
            conf = 0.5

        if not critical:
            fallback = self._basic_fallback(thesis, ticker, chunks)
            critical = fallback.critical_analysis

        if not biases:
            biases = [
                "Confirmation bias: podrías estar priorizando solo información favorable",
            ]
        if not recs:
            recs = [
                "Explora escenarios alternativos (optimista, base, pesimista)",
                "Revisa la calidad del negocio y la alineación del management",
            ]

        return ChallengeThesisResult(
            thesis=thesis,
            retrieved_chunks=chunks,
            critical_analysis=critical,
            identified_biases=biases,
            recommendations=recs,
            confidence_score=conf,
        )

    # ------------------------------------------------------------------
    # Llamadas a LLM
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        if not self.llm_available:
            raise RuntimeError("LLM no configurado")

        url = self.base_url
        if "chat/completions" not in url and not url.endswith("/"):
             url += "/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Eres un analista de inversiones racional."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return self._parse_json_from_text(content)

    # ------------------------------------------------------------------
    # Método público usado por /api/analysis/challenge
    # ------------------------------------------------------------------

    def challenge_thesis(
        self,
        thesis: str,
        ticker: str | None = None,
        top_k: int = 5,
    ) -> ChallengeThesisResult:
        """Desafía una tesis de inversión usando RAG + Llama."""
        query_text = thesis if not ticker else f"{thesis} ({ticker})"
        chunks = self._retrieve_chunks(query_text, top_k=top_k)

        prompt = self._build_prompt(thesis, ticker, chunks)

        # Intentar LLM
        if self.llm_available:
            try:
                data = self._call_llm(prompt)
                return self._normalize_output(thesis, ticker, chunks, data)
            except Exception as exc:
                print(f"[RAGService] Error con LLM, usando fallback básico: {exc}")

        # Fallback básico
        return self._basic_fallback(thesis, ticker, chunks)
