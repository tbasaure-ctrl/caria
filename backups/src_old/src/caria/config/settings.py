"""Configuración central de Caria.

Utiliza un enfoque jerárquico inspirado en Hydra para combinar configuraciones
base y overrides por entorno.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
import os
import re
from pathlib import Path
from typing import Any

import yaml


LOGGER = logging.getLogger("caria.config")


@dataclass(slots=True)
class Settings:
    """Representa la configuración cargada para una ejecución.

    Attributes:
        raw (dict[str, Any]): diccionario completo de configuración.
    """

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, base_path: Path, overrides: dict[str, Any] | None = None) -> "Settings":
        """Construye Settings a partir de un archivo YAML.

        Args:
            base_path: Ruta del archivo YAML principal.
            overrides: Diccionario con valores a sobreescribir.

        Returns:
            Instancia de Settings fusionando configuraciones.
        """

        LOGGER.debug("Cargando configuración base desde %s", base_path)
        with base_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        if overrides:
            LOGGER.debug("Aplicando overrides: %s", overrides)
            data = cls._merge_dicts(data, overrides)

        return cls(raw=data)

    @staticmethod
    def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(json.dumps(base))  # deep copy
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = Settings._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, *keys: str, default: Any = None) -> Any:
        """Obtiene un valor navegando por claves anidadas."""

        node = self._lookup_raw(keys)
        if node is None:
            return default
        expanded = self._expand(node, path_chain=[".".join(keys)] if keys else None)
        return expanded if expanded is not None else default

    def _lookup_raw(self, keys: tuple[str, ...]) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
            if node is None:
                return None
        return node

    PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")

    def _expand(self, value: Any, path_chain: list[str] | None = None) -> Any:
        if isinstance(value, str):
            return self._expand_string(value, path_chain=path_chain or [])
        if isinstance(value, list):
            return [self._expand(item, path_chain=path_chain) for item in value]
        if isinstance(value, dict):
            return {key: self._expand(item, path_chain=path_chain) for key, item in value.items()}
        return value

    def _expand_string(self, value: str, path_chain: list[str]) -> str:
        def replacement(match: re.Match[str]) -> str:
            token = match.group(1).strip()
            if token in path_chain:
                LOGGER.warning("Se detectó ciclo de referencias al resolver %s", token)
                return ""

            if token.startswith("oc.env:"):
                _, _, remainder = token.partition(":")
                env_key, _, default = remainder.partition(",")
                env_key = env_key.strip()
                default = default.strip() if default else ""
                return os.getenv(env_key, default)

            ref_keys = tuple(part.strip() for part in token.split(".") if part.strip())
            if not ref_keys:
                return ""

            raw_ref = self._lookup_raw(ref_keys)
            if raw_ref is None:
                LOGGER.warning("Placeholder %s no encontrado en configuración", token)
                return ""

            path_chain.append(token)
            resolved = self._expand(raw_ref, path_chain=path_chain)
            path_chain.pop()

            if isinstance(resolved, (dict, list)):
                return json.dumps(resolved)
            return str(resolved)

        expanded = self.PLACEHOLDER_RE.sub(replacement, value)
        if self.PLACEHOLDER_RE.search(expanded):
            return self._expand_string(expanded, path_chain=path_chain)
        return expanded

