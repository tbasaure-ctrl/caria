"""Script para validar el modelo HMM entrenado."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pickle
import numpy as np
import pandas as pd

def validate_hmm_model():
    """Valida el modelo HMM cargándolo y revisando sus parámetros."""
    model_path = Path(__file__).parent / "models" / "regime_hmm_model.pkl"

    print(f"Cargando modelo desde: {model_path}")
    print(f"Existe: {model_path.exists()}\n")

    if not model_path.exists():
        print("ERROR: Modelo no encontrado")
        return

    try:
        # Cargar modelo
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        print("=== MODELO HMM CARGADO EXITOSAMENTE ===\n")
        print(f"Keys en model_data: {list(model_data.keys())}\n")

        # Información básica
        print(f"N states: {model_data['n_states']}")
        print(f"Feature names: {model_data['feature_names']}")
        print(f"State labels: {model_data['state_labels']}\n")

        # Parámetros HMM
        hmm_model = model_data['model']
        print("=== PARÁMETROS HMM ===")
        print(f"\nStart probabilities (prior de estados):")
        for i, prob in enumerate(hmm_model.startprob_):
            label = model_data['state_labels'].get(i, f"State {i}")
            print(f"  {label}: {prob:.4f}")

        print(f"\nTransition matrix (probabilidades de transición):")
        print(f"  Shape: {hmm_model.transmat_.shape}")
        df_trans = pd.DataFrame(
            hmm_model.transmat_,
            index=[model_data['state_labels'].get(i, f"S{i}") for i in range(4)],
            columns=[model_data['state_labels'].get(i, f"S{i}") for i in range(4)],
        )
        print(df_trans)

        print(f"\nMeans por estado (características promedio):")
        print(f"  Shape: {hmm_model.means_.shape}")
        df_means = pd.DataFrame(
            hmm_model.means_,
            index=[model_data['state_labels'].get(i, f"S{i}") for i in range(4)],
            columns=model_data['feature_names'],
        )
        print(df_means)

        print(f"\n=== CONVERGENCIA ===")
        print(f"Convergió: {hmm_model.monitor_.converged}")
        print(f"Iteraciones ejecutadas: {hmm_model.monitor_.iter}")

        # Verificar predicciones
        pred_path = Path(__file__).parent / "data" / "silver" / "regime" / "hmm_regime_predictions.parquet"
        if pred_path.exists():
            print(f"\n=== PREDICCIONES ===")
            df_pred = pd.read_parquet(pred_path)
            print(f"Shape: {df_pred.shape}")
            print(f"Date range: {df_pred['date'].min()} to {df_pred['date'].max()}")
            print(f"\nDistribución de regímenes:")
            print(df_pred['regime'].value_counts())
            print(f"\nPromedio de confianza: {df_pred['confidence'].mean():.4f}")
            print(f"\nPrimeras 5 filas:")
            print(df_pred.head())
        else:
            print(f"\n⚠️ No se encontraron predicciones en {pred_path}")

        print("\n✅ Validación completada")

    except Exception as e:
        print(f"❌ ERROR al cargar modelo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    validate_hmm_model()
