"""Diagnóstico de overfitting: compara métricas train/val/test para todos los modelos."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
)

# Intentar importar PyTorch Lightning (opcional si hay problemas con DLLs)
try:
    import pytorch_lightning as pl
    HAS_PYTORCH = True
except (ImportError, OSError) as e:
    print(f"[WARNING] PyTorch Lightning no disponible: {e}")
    print("  Continuando solo con evaluación de modelos .pkl")
    HAS_PYTORCH = False
    pl = None

# Ajustar sys.path para importaciones
CURRENT_FILE = Path(__file__).resolve()
# BASE_DIR es el directorio notebooks (padre de scripts/)
def _find_base_dir() -> Path:
    """Encuentra el directorio base del proyecto (notebooks/)."""
    current = CURRENT_FILE
    # Si estamos en scripts/diagnostics/check_overfitting.py, subir 2 niveles
    if len(current.parts) >= 3 and current.parts[-3] == "scripts":
        base = current.parents[2]
    else:
        base = current.parents[1]
    
    # Verificar que estamos en el directorio correcto
    if not (base / "data" / "gold").exists() and not (base / "models").exists():
        # Intentar alternativas
        potential_bases = [
            current.parent.parent.parent,  # notebooks/
            Path.cwd(),  # Directorio actual
        ]
        for pb in potential_bases:
            if (pb / "data" / "gold").exists() or (pb / "models").exists():
                base = pb
                break
    return base

BASE_DIR = _find_base_dir()

POTENTIAL_SRC_DIRS = [
    BASE_DIR / "src",
    BASE_DIR / "caria" / "src",
]
for candidate in POTENTIAL_SRC_DIRS:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from caria.config.settings import Settings

if HAS_PYTORCH:
    try:
        from caria.models.training.datamodule import CariaDataModule
        from caria.models.training.workflow import SimpleFusionModel
    except Exception as e:
        print(f"[WARNING] No se pudieron importar módulos PyTorch: {e}")
        HAS_PYTORCH = False
        CariaDataModule = None
        SimpleFusionModel = None
else:
    CariaDataModule = None
    SimpleFusionModel = None


def parse_args() -> argparse.Namespace:
    # Calcular BASE_DIR antes de usarlo en defaults
    CURRENT_FILE = Path(__file__).resolve()
    _BASE_DIR = CURRENT_FILE.parents[2] if CURRENT_FILE.parts[-3] == "scripts" else CURRENT_FILE.parents[1]
    
    # Buscar config en ubicaciones posibles
    config_candidates = [
        _BASE_DIR / "configs" / "base.yaml",
        _BASE_DIR / "caria" / "configs" / "base.yaml",
    ]
    default_config = None
    for candidate in config_candidates:
        if candidate.exists():
            default_config = str(candidate.resolve())
            break
    if default_config is None:
        default_config = str(config_candidates[0])  # Usar el primero como fallback
    
    parser = argparse.ArgumentParser(description="Diagnóstico de overfitting")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Ruta al archivo YAML de configuración",
    )
    parser.add_argument(
        "--checkpoint",
        help="Ruta al checkpoint de SimpleFusionModel (.ckpt). Opcional.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_BASE_DIR / "artifacts" / "diagnostics"),
        help="Directorio para guardar reportes",
    )
    return parser.parse_args()


def evaluate_pkl_model(
    model_path: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    is_classifier: bool = False,
) -> dict[str, dict[str, float]]:
    """Evalúa un modelo .pkl en train/val/test."""
    model = joblib.load(model_path)
    results: dict[str, dict[str, float]] = {}

    for split_name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        if X.empty or y.empty:
            continue

        y_pred = model.predict(X)
        if is_classifier:
            y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred
            results[split_name] = {
                "accuracy": float(accuracy_score(y, y_pred)),
                "auc_roc": float(roc_auc_score(y, y_pred_proba)) if len(np.unique(y)) > 1 else 0.0,
            }
        else:
            results[split_name] = {
                "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                "mae": float(mean_absolute_error(y, y_pred)),
                "r2": float(r2_score(y, y_pred)),
            }

    return results


def evaluate_pytorch_model(
    checkpoint_path: Path,
    datamodule: CariaDataModule,
    settings: Settings,
) -> dict[str, dict[str, float]]:
    """Evalúa SimpleFusionModel en train/val/test."""
    if not HAS_PYTORCH:
        return {}
    
    model = SimpleFusionModel.load_from_checkpoint(
        str(checkpoint_path),
        input_dim=datamodule.get_feature_dim(),
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_checkpointing=False,
    )

    results: dict[str, dict[str, float]] = {}

    # Train
    train_results = trainer.validate(model=model, dataloaders=datamodule.train_dataloader())
    if train_results:
        results["train"] = {k.replace("val_", ""): float(v) for k, v in train_results[0].items()}

    # Validation
    val_results = trainer.validate(model=model, dataloaders=datamodule.val_dataloader())
    if val_results:
        results["val"] = {k.replace("val_", ""): float(v) for k, v in val_results[0].items()}

    # Test
    test_results = trainer.test(model=model, dataloaders=datamodule.test_dataloader())
    if test_results:
        results["test"] = {k.replace("test_", ""): float(v) for k, v in test_results[0].items()}

    return results


def calculate_gaps(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Calcula gaps entre splits."""
    gaps: dict[str, dict[str, float]] = {}

    if "train" in metrics and "val" in metrics:
        train_val_gaps = {}
        for metric in metrics["train"]:
            if metric in metrics["val"]:
                train_val_gaps[f"train_val_gap_{metric}"] = (
                    metrics["train"][metric] - metrics["val"][metric]
                )
        gaps["train_val"] = train_val_gaps

    if "val" in metrics and "test" in metrics:
        val_test_gaps = {}
        for metric in metrics["val"]:
            if metric in metrics["test"]:
                val_test_gaps[f"val_test_gap_{metric}"] = (
                    metrics["val"][metric] - metrics["test"][metric]
                )
        gaps["val_test"] = val_test_gaps

    if "train" in metrics and "test" in metrics:
        train_test_gaps = {}
        for metric in metrics["train"]:
            if metric in metrics["test"]:
                train_test_gaps[f"train_test_gap_{metric}"] = (
                    metrics["train"][metric] - metrics["test"][metric]
                )
        gaps["train_test"] = train_test_gaps

    return gaps


def prepare_pkl_features(
    df: pd.DataFrame, 
    model_path: Path | None = None,
    feature_config: dict | None = None,
    model_name: str | None = None,
    parquet_file_path: Path | None = None,
    all_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepara features y target para modelos .pkl."""
    # Intentar obtener features del modelo directamente
    expected_features = None
    if model_path and model_path.exists():
        try:
            model = joblib.load(model_path)
            # XGBoost y LightGBM tienen feature_names_in_
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            # También verificar get_booster para XGBoost
            elif hasattr(model, "get_booster"):
                try:
                    booster = model.get_booster()
                    if hasattr(booster, "feature_names"):
                        expected_features = booster.feature_names
                except:
                    pass
        except:
            pass
    
    # Si necesitamos leer más columnas del parquet, hacerlo ahora
    if expected_features and parquet_file_path and all_columns:
        cols_to_read = list(set(expected_features + ["target"]))
        cols_to_read = [c for c in cols_to_read if c in all_columns]
        
        missing_cols = [c for c in cols_to_read if c not in df.columns]
        if missing_cols:
            # Leer solo las columnas faltantes
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(parquet_file_path)
            df_additional = parquet_file.read(columns=missing_cols).to_pandas()
            # Combinar con lo que ya tenemos usando índice (asegurar mismo índice)
            df = pd.concat([df.reset_index(drop=True), df_additional.reset_index(drop=True)], axis=1)
            del df_additional
    
    # Si no se pueden obtener del modelo, usar feature_config
    if expected_features is None and feature_config:
        if model_name == "quality_model":
            expected_features = feature_config.get("quality_features", [])
        elif model_name == "valuation_model":
            expected_features = feature_config.get("valuation_features", [])
        elif model_name == "momentum_model":
            expected_features = feature_config.get("momentum_features", [])
        else:
            quality_features = feature_config.get("quality_features", [])
            valuation_features = feature_config.get("valuation_features", [])
            momentum_features = feature_config.get("momentum_features", [])
            expected_features = list(set(quality_features + valuation_features + momentum_features))
    
    # Si aún no tenemos features, usar todas las numéricas disponibles
    if expected_features is None:
        exclude_cols = {
            "ticker",
            "date",
            "target",
            "target_return_20d",
            "target_drawdown_prob",
            "features",
            "feature_columns",
            "period",
        }
        expected_features = [col for col in df.columns if col not in exclude_cols]
    
    # Filtrar solo las features que existen en el DataFrame
    available_features = [f for f in expected_features if f in df.columns]
    missing_features = [f for f in expected_features if f not in df.columns]
    
    if missing_features:
        print(f"  [WARNING] Features faltantes en datos: {missing_features[:5]}...")
    
    if not available_features:
        raise ValueError(f"No hay features disponibles. Esperadas: {expected_features[:10]}...")
    
    # Seleccionar solo las columnas necesarias y convertir a float32 para ahorrar memoria
    X = df[available_features].copy()
    # Usar fillna de forma más eficiente y convertir a float32
    for col in X.columns:
        if X[col].dtype in ['float64', 'float32']:
            X[col] = X[col].fillna(0.0).astype('float32')
        else:
            X[col] = X[col].fillna(0)
    
    y = df["target"].copy()
    return X, y


def main() -> None:
    args = parse_args()
    
    if HAS_PYTORCH and Settings:
        settings = Settings.from_yaml(Path(args.config))
    else:
        settings = None
        print("[INFO] Ejecutando sin PyTorch - solo modelos .pkl serán evaluados")

    # Cargar datos - rutas robustas
    if settings:
        gold_path = Path(settings.get("storage", "gold_path", default="data/gold"))
    else:
        gold_path = BASE_DIR / "data" / "gold"
    if not gold_path.is_absolute():
        gold_path = BASE_DIR / gold_path
    
    # Verificar que el directorio existe
    if not gold_path.exists():
        print(f"[ERROR] Directorio gold no encontrado: {gold_path}")
        print(f"  BASE_DIR: {BASE_DIR}")
        print(f"  Directorio actual: {Path.cwd()}")
        return
    
    # Verificar que los archivos existen
    train_file = gold_path / "train.parquet"
    val_file = gold_path / "val.parquet"
    test_file = gold_path / "test.parquet"
    
    for f in [train_file, val_file, test_file]:
        if not f.exists():
            print(f"[ERROR] Archivo no encontrado: {f}")
            return
    
    print(f"[INFO] Cargando datos desde: {gold_path}")
    print(f"  train.parquet: {train_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  val.parquet: {val_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  test.parquet: {test_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Leer archivos - primero obtener metadata para saber qué columnas leer
    # Guardamos metadata en variables separadas (no en DataFrames)
    parquet_metadata = {}
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(train_file)
        parquet_metadata["all_columns"] = parquet_file.schema.names
        parquet_metadata["train_file"] = train_file
        parquet_metadata["val_file"] = val_file
        parquet_metadata["test_file"] = test_file
    except Exception as e:
        print(f"[WARNING] No se pudo leer metadata: {e}")
        parquet_metadata = None
    
    # Determinar qué columnas necesitamos para todos los modelos
    # Primero cargar los modelos para saber qué features necesitan
    models_dir = BASE_DIR / "models"
    all_needed_cols = set(["target", "date", "roic"])  # roic para quality_model labels
    
    for model_name in ["quality_model", "valuation_model", "momentum_model"]:
        model_path = models_dir / f"{model_name}.pkl"
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                if hasattr(model, "feature_names_in_"):
                    all_needed_cols.update(model.feature_names_in_)
                elif hasattr(model, "get_booster"):
                    try:
                        booster = model.get_booster()
                        if hasattr(booster, "feature_names"):
                            all_needed_cols.update(booster.feature_names)
                    except:
                        pass
            except:
                pass
    
    # Filtrar solo columnas que existen en el parquet
    if parquet_metadata:
        all_needed_cols = [c for c in all_needed_cols if c in parquet_metadata["all_columns"]]
    else:
        all_needed_cols = list(all_needed_cols)
    
    print(f"  Leyendo {len(all_needed_cols)} columnas necesarias...")
    try:
        train_df = pd.read_parquet(train_file, columns=all_needed_cols)
        val_df = pd.read_parquet(val_file, columns=all_needed_cols)
        test_df = pd.read_parquet(test_file, columns=all_needed_cols)
        print(f"  [OK] train cargado: {len(train_df)} filas, {len(train_df.columns)} columnas")
        print(f"  [OK] val cargado: {len(val_df)} filas, {len(val_df.columns)} columnas")
        print(f"  [OK] test cargado: {len(test_df)} filas, {len(test_df.columns)} columnas")
    except Exception as e:
        print(f"[ERROR] No se pudo leer archivos parquet: {e}")
        print(f"  Intentando leer solo columnas esenciales...")
        # Fallback: leer solo lo esencial
        essential_cols = ["target", "date"]
        train_df = pd.read_parquet(train_file, columns=essential_cols)
        val_df = pd.read_parquet(val_file, columns=essential_cols)
        test_df = pd.read_parquet(test_file, columns=essential_cols)

    # Cargar feature config si existe
    feature_config_path = BASE_DIR / "models" / "feature_config.pkl"
    feature_config = None
    if feature_config_path.exists():
        feature_config = joblib.load(feature_config_path)

    report: dict[str, any] = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }

    # Evaluar modelos .pkl
    models_dir = BASE_DIR / "models"
    for model_name in ["quality_model", "valuation_model", "momentum_model"]:
        model_path = models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            continue

        print(f"\nEvaluando {model_name}...")

        # Preparar features usando el modelo para obtener las features esperadas
        # Ya tenemos todas las columnas necesarias cargadas, así que no necesitamos lectura bajo demanda
        X_train, y_train = prepare_pkl_features(
            train_df.copy(), 
            model_path=model_path, 
            feature_config=feature_config, 
            model_name=model_name,
        )
        X_val, y_val = prepare_pkl_features(
            val_df.copy(), 
            model_path=model_path, 
            feature_config=feature_config, 
            model_name=model_name,
        )
        X_test, y_test = prepare_pkl_features(
            test_df.copy(), 
            model_path=model_path, 
            feature_config=feature_config, 
            model_name=model_name,
        )
        
        print(f"  Features usadas: {len(X_train.columns)} ({', '.join(X_train.columns[:5])}...)")

        # Determinar si es clasificador (quality y momentum son clasificadores)
        is_classifier = model_name in ["quality_model", "momentum_model"]

        if is_classifier:
            # Para clasificadores, necesitamos recrear labels
            if model_name == "quality_model":
                # "roic" ya debería estar cargado
                train_df_copy = train_df.copy()
                train_df_copy["roic_percentile"] = train_df_copy.groupby("date")["roic"].rank(pct=True)
                train_df_copy["is_quality"] = (train_df_copy["roic_percentile"] > 0.80).astype(int)
                y_train = train_df_copy["is_quality"]

                val_df_copy = val_df.copy()
                val_df_copy["roic_percentile"] = val_df_copy.groupby("date")["roic"].rank(pct=True)
                val_df_copy["is_quality"] = (val_df_copy["roic_percentile"] > 0.80).astype(int)
                y_val = val_df_copy["is_quality"]

                test_df_copy = test_df.copy()
                test_df_copy["roic_percentile"] = test_df_copy.groupby("date")["roic"].rank(pct=True)
                test_df_copy["is_quality"] = (test_df_copy["roic_percentile"] > 0.80).astype(int)
                y_test = test_df_copy["is_quality"]
            elif model_name == "momentum_model":
                y_train = (train_df["target"] > 0).astype(int)
                y_val = (val_df["target"] > 0).astype(int)
                y_test = (test_df["target"] > 0).astype(int)

        metrics = evaluate_pkl_model(
            model_path, X_train, y_train, X_val, y_val, X_test, y_test, is_classifier=is_classifier
        )
        gaps = calculate_gaps(metrics)

        report["models"][model_name] = {
            "metrics": metrics,
            "gaps": gaps,
            "is_classifier": is_classifier,
        }

    # Evaluar SimpleFusionModel si se proporciona checkpoint
    if args.checkpoint:
        if not HAS_PYTORCH:
            print("\n[WARNING] PyTorch no disponible - omitiendo evaluación de SimpleFusionModel")
        else:
            print("\nEvaluando SimpleFusionModel...")
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = BASE_DIR / checkpoint_path

            if checkpoint_path.exists() and settings:
                try:
                    datamodule = CariaDataModule(data_root=gold_path, settings=settings)
                    datamodule.setup(stage="test")

                    metrics = evaluate_pytorch_model(checkpoint_path, datamodule, settings)
                    gaps = calculate_gaps(metrics)

                    report["models"]["simple_fusion"] = {
                        "metrics": metrics,
                        "gaps": gaps,
                        "is_classifier": False,
                    }
                except Exception as e:
                    print(f"  [ERROR] No se pudo evaluar SimpleFusionModel: {e}")
            else:
                print(f"  [WARNING] Checkpoint no encontrado o settings no disponibles")

    # Guardar reporte
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d")
    report_path = output_dir / f"overfitting_report_{timestamp_str}.json"

    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReporte guardado en: {report_path}")

    # Imprimir resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE OVERFITTING")
    print("=" * 60)
    for model_name, model_data in report["models"].items():
        print(f"\n{model_name}:")
        metrics = model_data["metrics"]
        gaps = model_data["gaps"]

        for split in ["train", "val", "test"]:
            if split in metrics:
                print(f"  {split.upper()}:")
                for metric, value in metrics[split].items():
                    print(f"    {metric}: {value:.4f}")

        if "train_val" in gaps:
            print("  GAPS train-val:")
            for gap_name, gap_value in gaps["train_val"].items():
                print(f"    {gap_name}: {gap_value:.4f}")

        if "val_test" in gaps:
            print("  GAPS val-test:")
            for gap_name, gap_value in gaps["val_test"].items():
                print(f"    {gap_name}: {gap_value:.4f}")


if __name__ == "__main__":
    main()

