"""Generación de reportes estructurados."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ReportSection:
    title: str
    body: str


@dataclass
class Report:
    ticker: str
    sections: list[ReportSection]
    sources: list[dict[str, Any]]


def build_report(ticker: str, insights: list[str], sources: list[dict[str, Any]]) -> Report:
    sections = [ReportSection(title=f"Insight {idx+1}", body=insight) for idx, insight in enumerate(insights)]
    return Report(ticker=ticker, sections=sections, sources=sources)

