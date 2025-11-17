"""Conexión a bases de datos relacionales y vectoriales."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

LOGGER = logging.getLogger("caria.data_access.db")


class DatabaseFactory:
    """Administra instancias de SQLAlchemy Engine reutilizables."""

    def __init__(self) -> None:
        self._engines: dict[str, Engine] = {}

    def get_engine(self, connection_uri: str) -> Engine:
        if connection_uri not in self._engines:
            LOGGER.info("Creando engine de base de datos: %s", connection_uri)
            self._engines[connection_uri] = create_engine(connection_uri, future=True)
        return self._engines[connection_uri]


factory = DatabaseFactory()


@contextmanager
def session_scope(connection_uri: str) -> Iterator[Engine]:
    """Provide a transactional scope around a series of operations."""

    engine = factory.get_engine(connection_uri)
    connection = engine.connect()
    trans = connection.begin()
    try:
        yield connection
        trans.commit()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Error en transacción, realizando rollback")
        trans.rollback()
        raise exc
    finally:
        connection.close()

