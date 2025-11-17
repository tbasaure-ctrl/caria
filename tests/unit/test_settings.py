from pathlib import Path

from caria.config.settings import Settings


def test_settings_loads_yaml(tmp_path: Path) -> None:
    content = "test: {value: 1}"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")
    settings = Settings.from_yaml(config_path)
    assert settings.get("test", "value") == 1

