"""Persistencia del historial de ejecuciones de pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import MetaData, Table, Column, String, DateTime, JSON

from caria.data_access.db import factory


metadata = MetaData()

jobs_log = Table(
    "jobs_log",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("job_name", String(80), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("status", String(20), nullable=False),
    Column("details", JSON, nullable=True),
)


@dataclass(slots=True)
class JobRecord:
    job_name: str
    job_id: str
    started_at: datetime
    status: str
    details: dict[str, Any] | None = None


def ensure_tables(connection_uri: str) -> None:
    engine = factory.get_engine(connection_uri)
    metadata.create_all(engine)

