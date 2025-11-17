import yaml


def test_ingestion_config_loads() -> None:
    with open("configs/pipelines/ingestion.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert "sources" in config

