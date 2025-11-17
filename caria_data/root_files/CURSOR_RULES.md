# CURSOR RULES — Wise Adviser Project

Reglas específicas para el IDE Cursor al trabajar en este proyecto.

## 1. Indexado de Carpetas Prioritarias

Cursor debe indexar automáticamente:

### Prioridad ALTA (siempre)
- `AI_CONTEXT.md`
- `PERSISTENT_CONTEXT.md`
- `data_schema/`
- `src/valuation/`
- `src/models/`
- `scripts/`

### Prioridad MEDIA (según tarea)
- `notebooks/` — Para análisis/debugging
- `tests/` — Para desarrollo con TDD
- `infrastructure/` — Para setup/deploy

### Prioridad BAJA (bajo demanda)
- `raw/` — Solo metadata, no contenido
- `models/` — Solo checkpoints recientes
- `experiments/` — Solo manifests actuales

## 2. Uso de @Folders en Prompts

### Patrón Recomendado
```
@Folders: {carpetas_relevantes}
{tu_prompt}
```

### Ejemplos

**Para análisis de valuación:**
```
@Folders: src/valuation/, data_schema/
Analiza AAPL con DCF considerando fair value y MOS
```

**Para debugging de modelo:**
```
@Folders: src/models/, notebooks/
El encoder macro no converge. Debug loss function.
```

**Para refactoring:**
```
@Folders: src/features/, tests/features/
Refactoriza macro_features.py para mejor testability
```

## 3. Llamadas al MCP Server

### Siempre Incluir
```python
{
  "query": "tu_query_aquí",
  "top_k": 5,
  "filters": {
    "themes": ["valuation", "discipline"],
    "index_version": "v1"  # ← CRÍTICO para consistencia
  }
}
```

### Filters Recomendados por Caso

**Valuación:**
```python
filters = {
    "themes": ["valuation", "margin_of_safety"],
    "source": "Graham",  # o "Buffett", "Fisher"
}
```

**Psicología de Masas:**
```python
filters = {
    "themes": ["behavioral", "psychology"],
    "context": "crisis"  # o "mania", "bubble"
}
```

**Régimen Macro:**
```python
filters = {
    "themes": ["macro", "cycles"],
    "source": "Dalio"  # o "Marks"
}
```

## 4. No Ejecutar Sin Confirmar

### Scripts que requieren confirmación explícita:

❌ **NO ejecutar automáticamente:**
- `scripts/01_download_data.py` — Consume API calls
- `scripts/02_train_model.py` — Consume GPU/tiempo
- `scripts/embed_and_index.py` — Modifica índice vectorial
- Cualquier script que modifique DB

✅ **OK ejecutar sin confirmar:**
- Tests (`pytest`)
- Linters (`black`, `isort`, `flake8`)
- Type checks (`mypy`)
- Lectura de datos (read-only)

### Patrón de Confirmación
```
Cursor: "Necesito ejecutar {script} para {razón}. 
         Esto va a {efecto_secundario}. 
         ¿Procedo?"
         
Usuario: "Sí" / "No"
```

## 5. Proponer Cambios Estructurales

### Workflow para Refactorings

1. **Analizar** código actual
2. **Proponer** cambios con justificación
3. **Mostrar** diff/patch
4. **Incluir** tests para nuevos cambios
5. **Esperar** confirmación del usuario

### Ejemplo de Propuesta

```
📋 Propuesta: Refactor de src/features/macro_features.py

🎯 Objetivo:
- Separar data loading de feature computation
- Mejorar testability (inject dependencies)
- Reducir acoplamiento con FMP API

🔧 Cambios:
1. Crear MacroDataLoader class
2. Extraer compute_yield_curve_slope a pure function
3. Add dependency injection para API client

📝 Diff:
[mostrar diff aquí]

🧪 Tests:
- tests/features/test_macro_features.py (new)
- tests/integration/test_macro_pipeline.py (updated)

¿Procedo con el refactor?
```

## 6. Reglas de Código

### Type Hints Obligatorios

✅ Correcto:
```python
def calculate_dcf(
    fcf: list[float],
    discount_rate: float,
    terminal_growth: float = 0.025
) -> dict[str, float]:
    """Calculate DCF valuation."""
    ...
```

❌ Incorrecto:
```python
def calculate_dcf(fcf, discount_rate, terminal_growth=0.025):
    ...
```

### Docstrings (Google Style)

```python
def calculate_margin_of_safety(fair_value: float, current_price: float) -> float:
    """Calculate margin of safety as percentage.
    
    Args:
        fair_value: Estimated fair value per share
        current_price: Current market price
        
    Returns:
        Margin of safety as decimal (e.g., 0.18 for 18%)
        Positive = undervalued, Negative = overvalued
        
    Example:
        >>> calculate_margin_of_safety(195.0, 165.0)
        0.18
    """
    return (fair_value - current_price) / fair_value
```

### Imports con isort

```python
# Standard library
import os
from datetime import datetime
from typing import Dict, List

# Third-party
import numpy as np
import pandas as pd
import torch

# Local
from src.config import Config
from src.valuation.dcf import calculate_dcf
```

## 7. Testing Guidelines

### Estructura de Tests

```python
# tests/test_valuation_engine.py

import pytest
from src.valuation.valuation_engine import ValuationEngine

class TestValuationEngine:
    @pytest.fixture
    def engine(self):
        return ValuationEngine(ticker="TEST", api_key="fake_key")
    
    def test_dcf_with_known_values(self, engine):
        """Test DCF calculation with known inputs."""
        # Arrange
        fcf = [100, 105, 110]
        discount_rate = 0.10
        
        # Act
        result = engine._calculate_dcf_value(fcf, discount_rate)
        
        # Assert
        assert result > 0
        assert isinstance(result, float)
    
    def test_margin_of_safety_undervalued(self, engine):
        """MOS should be positive when undervalued."""
        fair_value = 200.0
        current_price = 150.0
        
        mos = engine._calculate_margin_of_safety(fair_value, current_price)
        
        assert mos > 0
        assert mos == pytest.approx(0.25)
```

### Coverage Mínima

- **Critical modules**: >90% (valuation, models, features)
- **Scripts**: >70%
- **Utils**: >80%

## 8. Uso de Notebooks

### Naming Convention

- `eda_*.ipynb` — Exploratory Data Analysis
- `debug_*.ipynb` — Debugging sessions
- `experiment_*.ipynb` — Experimentos de modelo
- `viz_*.ipynb` — Visualizaciones

### Estructura Recomendada

```python
# %% [markdown]
# # Título del Análisis
# Descripción breve

# %% Imports
import pandas as pd
...

# %% Load Data
data = pd.read_parquet("...")

# %% Analysis
...

# %% Visualization
...

# %% Conclusions
# - Finding 1
# - Finding 2
```

## 9. Commit Messages (si Cursor hace commits)

### Formato

```
type(scope): short description

Longer description if needed

- Detail 1
- Detail 2
```

### Types
- `feat`: Nueva funcionalidad
- `fix`: Bug fix
- `refactor`: Refactoring sin cambio funcional
- `test`: Agregar o modificar tests
- `docs`: Documentación
- `chore`: Tareas de mantenimiento

### Ejemplos

```
feat(valuation): add reverse DCF calculation

Implements implied growth rate calculation by working 
backwards from current price.

- Added reverse_dcf method to ValuationEngine
- Added tests in test_valuation_engine.py
- Updated docs with usage example
```

```
fix(embeddings): correct chunk overlap calculation

Fixed off-by-one error in chunk_text function that was
causing overlapping chunks to miss tokens.

Closes #42
```

## 10. Debugging Workflow

### Paso 1: Identificar el Problema
```
@Folders: {carpeta_relevante}
El script {X} falla con error: {error_message}
```

### Paso 2: Explorar Contexto
- Leer logs en `models/logs/`
- Check state en notebooks
- Verificar inputs/outputs

### Paso 3: Proponer Fix
- Mostrar causa raíz
- Proponer solución
- Incluir test que reproduzca el bug

### Paso 4: Validar
- Ejecutar test
- Verificar fix no rompe nada más
- Update docs si necesario

---

## Resumen: Cursor Checklist

Antes de cada sesión:
- [ ] Leer `AI_CONTEXT.md`
- [ ] Leer `PERSISTENT_CONTEXT.md`
- [ ] Indexar carpetas prioritarias

Al modificar código:
- [ ] Usar @Folders apropiados
- [ ] Type hints en funciones nuevas
- [ ] Docstrings en funciones públicas
- [ ] Tests para nuevas features
- [ ] No ejecutar scripts destructivos sin confirmar

Al refactorizar:
- [ ] Proponer cambios con justificación
- [ ] Mostrar diff
- [ ] Incluir tests
- [ ] Esperar confirmación

Al debuggear:
- [ ] Identificar problema claramente
- [ ] Explorar contexto con notebooks/logs
- [ ] Proponer fix con test
- [ ] Validar no se rompa nada más

---

**Estas reglas maximizan la efectividad de Cursor en este proyecto.**

Última actualización: 2025-01-07
