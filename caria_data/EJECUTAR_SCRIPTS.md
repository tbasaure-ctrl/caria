# 🎯 Cómo Ejecutar Scripts - Guía Visual

## ✅ FORMA CORRECTA (Siempre funciona)

```powershell
# Paso 1: Ir al directorio caria_data
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data

# Paso 2: Ejecutar script con path relativo
python scripts/orchestration/run_regime_hmm.py
```

## 📍 Ubicación Correcta

```
C:\key\wise_adviser_cursor_context\notebooks\caria_data\  ← AQUÍ debes estar
├── src/
│   └── caria/          ← Módulo Python
├── scripts/
│   └── orchestration/
│       └── run_regime_hmm.py  ← Scripts aquí
└── configs/
    └── base.yaml       ← Configs aquí
```

## 🚀 Comandos Listos para Copiar

### Entrenar Sistema I (HMM Régimen)

**Requisito previo**: Asegúrate de tener datos macro en `data/silver/macro/fred_data.parquet`

```powershell
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data
python scripts/orchestration/run_regime_hmm.py
```

**Nota**: El script usa `fred_data.parquet` por defecto. El HMM calculará automáticamente las features necesarias (yield_curve_slope, sentiment_score, etc.) desde los datos FRED.

**Salida**: 
- Modelo entrenado: `models/regime_hmm_model.pkl`
- Predicciones históricas: `data/silver/regime/hmm_regime_predictions.parquet`

### Con configuración personalizada

```powershell
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data
python scripts/orchestration/run_regime_hmm.py --config configs/base.yaml --pipeline-config configs/pipelines/regime_hmm.yaml
```

## 🔍 Verificación Antes de Ejecutar

Ejecuta esto para verificar que estás en el lugar correcto:

```powershell
# Debe mostrar: C:\key\wise_adviser_cursor_context\notebooks\caria_data
pwd

# Debe retornar True:
Test-Path src/caria
Test-Path configs/base.yaml
Test-Path scripts/orchestration/run_regime_hmm.py
```

## ❌ Errores y Soluciones

### Error 1: `ModuleNotFoundError: No module named 'caria'`

**Causa**: No estás en `caria_data/` o ejecutaste desde otro directorio.

**Solución**:
```powershell
# Asegúrate de estar aquí:
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data

# Luego ejecuta:
python scripts/orchestration/run_regime_hmm.py
```

### Error 2: `FileNotFoundError: configs/base.yaml`

**Causa**: Ejecutaste desde un directorio diferente.

**Solución**: Siempre ejecuta desde `caria_data/`:
```powershell
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data
python scripts/orchestration/run_regime_hmm.py
```

## 📝 Notas Importantes

1. **SIEMPRE** ejecuta desde `caria_data/`
2. **SIEMPRE** usa paths relativos (`scripts/orchestration/...`)
3. Los scripts ahora configuran automáticamente los paths de Python
4. Los paths de configuración son relativos a `caria_data/`

## 🎓 Por Qué Funciona Ahora

El script `run_regime_hmm.py` ahora:
1. Detecta automáticamente su ubicación
2. Calcula `BASE_DIR` (caria_data/)
3. Agrega `src/` al PYTHONPATH
4. Resuelve paths de configuración relativos a BASE_DIR

**No necesitas configurar nada manualmente**, solo ejecuta desde `caria_data/`.

