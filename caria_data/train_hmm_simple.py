"""Script simple para entrenar HMM sin Prefect."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from caria.models.regime.hmm_regime_detector import HMMRegimeDetector

# Cargar datos macro
print("Cargando datos macro...")
df = pd.read_parquet("data/silver/macro/fred_data.parquet")
print(f"Cargados {len(df)} observaciones")

# Filtrar por período
df = df[(df["date"] >= "1990-01-01") & (df["date"] <= "2024-11-30")]
print(f"Filtrado a {len(df)} observaciones (1990-2024)")

# Entrenar HMM
print("\nEntrenando HMM...")
detector = HMMRegimeDetector(n_states=4, n_iter=100)
detector.fit(df)

# Guardar modelo
print("\nGuardando modelo...")
detector.save("models/regime_hmm_model.pkl")

# Predecir regímenes históricos
print("\nPrediciendo regímenes históricos...")
predictions = detector.predict_historical_regimes(df)
predictions.to_parquet("data/silver/regime/hmm_regime_predictions.parquet", index=False)

print(f"\n✅ Completado!")
print(f"Modelo guardado en: models/regime_hmm_model.pkl")
print(f"Predicciones guardadas en: data/silver/regime/hmm_regime_predictions.parquet")
print(f"\nDistribución de regímenes:")
print(predictions["regime"].value_counts())
print(f"\nConfianza promedio: {predictions['confidence'].mean():.2f}")
