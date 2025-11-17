from pathlib import Path

import pytest

from caria.ingestion.registry import build_source


def test_build_source_unknown(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_source("unknown", output_dir=tmp_path)

