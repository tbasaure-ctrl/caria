"""Genera reporte comparativo de métricas antes/después de regularización."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import joblib
import pandas as pd

# Ajustar sys.path
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parents[2]
POTENTIAL_SRC_DIRS = [
    BASE_DIR / "src",
    BASE_DIR / "caria" / "src",
]
for candidate in POTENTIAL_SRC_DIRS:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from caria.config.settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera reporte de regularización")
    parser.add_argument(
        "--config",
        default=str((BASE_DIR / "caria" / "configs" / "base.yaml").resolve()),
        help="Ruta al archivo YAML de configuración",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "artifacts" / "diagnostics"),
        help="Directorio para guardar reporte",
    )
    parser.add_argument(
        "--compare-overfitting-report",
        help="Ruta al reporte de overfitting anterior (JSON)",
    )
    return parser.parse_args()


def evaluate_model_on_test(
    model_path: Path, X_test: pd.DataFrame, y_test: pd.Series, is_classifier: bool = False
) -> dict[str, float]:
    """Evalúa un modelo en test."""
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        accuracy_score,
        roc_auc_score,
    )
    import numpy as np

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    if is_classifier:
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "auc_roc": float(roc_auc_score(y_test, y_pred_proba)) if len(np.unique(y_test)) > 1 else 0.0,
        }
    else:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("REPORTE DE REGULARIZACIÓN")
    print("=" * 60)

    # Cargar datos de test
    gold_path = BASE_DIR / "data" / "gold"
    test_df = pd.read_parquet(gold_path / "test.parquet")

    # Cargar feature config
    feature_config_path = BASE_DIR / "models" / "feature_config.pkl"
    if feature_config_path.exists():
        feature_config = joblib.load(feature_config_path)
        quality_features = feature_config.get("quality_features", [])
        valuation_features = feature_config.get("valuation_features", [])
        momentum_features = feature_config.get("momentum_features", [])
    else:
        print("[WARNING] feature_config.pkl no encontrado")
        return

    # Preparar features de test
    X_test_quality = test_df[[f for f in quality_features if f in test_df.columns]].fillna(0)
    X_test_valuation = test_df[[f for f in valuation_features if f in test_df.columns]].fillna(0)
    X_test_momentum = test_df[[f for f in momentum_features if f in test_df.columns]].fillna(0)

    y_test = test_df["target"]

    # Preparar labels para clasificadores
    test_df["roic_percentile"] = test_df.groupby("date")["roic"].rank(pct=True)
    test_df["is_quality"] = (test_df["roic_percentile"] > 0.80).astype(int)
    y_test_quality = test_df["is_quality"]

    test_df["is_undervalued"] = 0
    if "priceToBookRatio" in test_df.columns:
        test_df["pb_rank"] = test_df.groupby("date")["priceToBookRatio"].rank(pct=True)
        test_df.loc[test_df["pb_rank"] < 0.30, "is_undervalued"] = 1
    y_test_valuation = test_df["is_undervalued"]

    y_test_momentum = (y_test > 0).astype(int)

    # Evaluar modelos originales
    print("\n[1/3] Evaluando modelos originales...")
    models_dir = BASE_DIR / "models"
    original_metrics = {}

    for model_name, is_classifier, X_test, y_test_label in [
        ("quality_model", True, X_test_quality, y_test_quality),
        ("valuation_model", True, X_test_valuation, y_test_valuation),
        ("momentum_model", True, X_test_momentum, y_test_momentum),
    ]:
        model_path = models_dir / f"{model_name}.pkl"
        if model_path.exists():
            metrics = evaluate_model_on_test(model_path, X_test, y_test_label, is_classifier)
            original_metrics[model_name] = metrics
            print(f"  {model_name}: {metrics}")

    # Evaluar modelos regularizados
    print("\n[2/3] Evaluando modelos regularizados...")
    regularized_metrics = {}

    for model_name, is_classifier, X_test, y_test_label in [
        ("regularized_quality_model", True, X_test_quality, y_test_quality),
        ("regularized_valuation_model", True, X_test_valuation, y_test_valuation),
        ("regularized_momentum_model", True, X_test_momentum, y_test_momentum),
    ]:
        model_path = models_dir / f"{model_name}.pkl"
        if model_path.exists():
            metrics = evaluate_model_on_test(model_path, X_test, y_test_label, is_classifier)
            regularized_metrics[model_name] = metrics
            print(f"  {model_name}: {metrics}")

    # Comparar
    print("\n[3/3] Comparando métricas...")
    comparison = {}

    for model_name in original_metrics.keys():
        if model_name not in regularized_metrics:
            continue

        orig = original_metrics[model_name]
        reg = regularized_metrics[model_name.replace("_model", "_regularized_model")]

        comparison[model_name] = {}
        for metric in orig.keys():
            if metric in reg:
                improvement = reg[metric] - orig[metric]
                pct_change = (improvement / orig[metric] * 100) if orig[metric] != 0 else 0
                comparison[model_name][metric] = {
                    "original": orig[metric],
                    "regularized": reg[metric],
                    "improvement": improvement,
                    "pct_change": pct_change,
                }

    # Generar reporte
    report = {
        "timestamp": datetime.now().isoformat(),
        "original_metrics": original_metrics,
        "regularized_metrics": regularized_metrics,
        "comparison": comparison,
        "summary": {
            "models_evaluated": len(original_metrics),
            "regularized_models_evaluated": len(regularized_metrics),
        },
    }

    # Guardar JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d")
    json_path = output_dir / f"regularization_report_{timestamp_str}.json"

    with json_path.open("w") as f:
        json.dump(report, f, indent=2)

    # Generar markdown
    md_path = output_dir / f"regularization_report_{timestamp_str}.md"
    with md_path.open("w") as f:
        f.write("# Reporte de Regularización\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Resumen\n\n")
        f.write(f"- Modelos originales evaluados: {len(original_metrics)}\n")
        f.write(f"- Modelos regularizados evaluados: {len(regularized_metrics)}\n\n")

        f.write("## Comparación de Métricas\n\n")
        for model_name, comp in comparison.items():
            f.write(f"### {model_name}\n\n")
            f.write("| Métrica | Original | Regularizado | Mejora | % Cambio |\n")
            f.write("|---------|----------|--------------|--------|----------|\n")
            for metric, values in comp.items():
                f.write(
                    f"| {metric} | {values['original']:.4f} | {values['regularized']:.4f} | "
                    f"{values['improvement']:+.4f} | {values['pct_change']:+.2f}% |\n"
                )
            f.write("\n")

    print(f"\n[OK] Reporte guardado:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")

    print("\n" + "=" * 60)
    print("[COMPLETADO] REPORTE DE REGULARIZACIÓN")
    print("=" * 60)


if __name__ == "__main__":
    main()





