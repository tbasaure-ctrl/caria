# Protocolo MLOps: Validación Cruzada Purgada y Embargada

## ¿Por qué es necesario?

En datos financieros, la validación cruzada estándar introduce dos problemas críticos:

1. **Look-Ahead Bias**: Si una observación de train tiene un target que se superpone temporalmente con el test, estamos usando información del futuro para entrenar.
2. **Autocorrelación**: Las observaciones inmediatamente posteriores al test están correlacionadas con el test, inflando artificialmente las métricas.

## Solución: Purging y Embargo

### Purging
Elimina observaciones de train cuyas etiquetas se superponen temporalmente con test.

**Ejemplo:**
- Test: 2020-01-01 a 2020-01-31 (target: retorno 20 días = hasta 2020-02-20)
- Train debe excluir observaciones con target que se superponga con 2020-01-01 a 2020-02-20

### Embargo
Elimina observaciones inmediatamente posteriores al test para evitar autocorrelación.

**Ejemplo:**
- Test: 2020-01-01 a 2020-01-31
- Embargo de 1 día: excluir 2020-02-01 del siguiente fold

## Implementación

### Uso básico

```python
from caria.evaluation.purged_cv import PurgedKFold
import pandas as pd

# Datos con columna 'date'
X = features_df
y = targets_df
dates = features_df['date']

# Crear validador
cv = PurgedKFold(n_splits=5, purge_days=1, embargo_days=1)

# Usar en cross-validation
for train_idx, test_idx in cv.split(X, y, groups=dates):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Entrenar y evaluar modelo
```

### Configuración recomendada

- **Purge days**: 1-5 días (depende del horizonte del target)
- **Embargo days**: 1-3 días
- **Número de splits**: 3-5 (más splits = menos datos por fold)

## Impacto en Métricas

**Importante**: Las métricas con purged CV serán más bajas pero reflejarán rendimiento real.

- Métricas infladas (sin purging): R² = 0.85, Accuracy = 90%
- Métricas realistas (con purging): R² = 0.35, Accuracy = 65%

Esto es **esperado y correcto**. Las métricas más bajas indican que el modelo no está usando información del futuro.

## Aplicación

Este protocolo debe usarse para:
- ✅ Sistema I (HMM Régimen)
- ✅ Sistema III (Factores)
- ✅ Sistema IV (Valuación)
- ❌ Sistema II (RAG) - No aplica (no es modelo predictivo)

## Referencias

- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Machine Learning for Asset Managers" - Marcos López de Prado

