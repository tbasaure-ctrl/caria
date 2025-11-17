# Guía de Ejecución de Scripts

## Estructura de Directorios

```
caria_data/
├── src/
│   └── caria/          # Módulo principal (aquí están todos los sistemas)
├── scripts/
│   ├── orchestration/  # Scripts de pipelines
│   ├── diagnostics/    # Scripts de diagnóstico
│   └── ...
├── configs/            # Archivos de configuración
└── data/               # Datos (silver, gold, etc.)
```

## Cómo Ejecutar Scripts

### Opción 1: Desde el directorio `caria_data/` (RECOMENDADO)

```powershell
# Cambiar al directorio base
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data

# Ejecutar script con path relativo
python scripts/orchestration/run_regime_hmm.py
```

### Opción 2: Usando PYTHONPATH

```powershell
# Desde cualquier directorio
$env:PYTHONPATH = "C:\key\wise_adviser_cursor_context\notebooks\caria_data\src"
python C:\key\wise_adviser_cursor_context\notebooks\caria_data\scripts\orchestration\run_regime_hmm.py
```

### Opción 3: Usando módulo Python

```powershell
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data
python -m scripts.orchestration.run_regime_hmm
```

## Scripts Disponibles

### Orchestration (Pipelines)

- `scripts/orchestration/run_regime_hmm.py` - Entrenar Sistema I (HMM Régimen)
- Otros scripts de pipelines...

### Diagnostics

- `scripts/diagnostics/check_overfitting.py` - Diagnosticar overfitting

## Configuración de Paths

Los scripts automáticamente configuran `sys.path` para encontrar el módulo `caria`. Si tienes problemas:

1. Asegúrate de estar en el directorio `caria_data/`
2. Verifica que existe `src/caria/`
3. Si persiste, ejecuta desde `caria_data/` con path relativo

## Ejemplo Completo

```powershell
# 1. Ir al directorio base
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data

# 2. Verificar estructura
ls src/caria  # Debe mostrar los módulos

# 3. Ejecutar script
python scripts/orchestration/run_regime_hmm.py --config configs/base.yaml --pipeline-config configs/pipelines/regime_hmm.yaml
```

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'caria'`

**Solución**: Ejecuta desde `caria_data/`:
```powershell
cd C:\key\wise_adviser_cursor_context\notebooks\caria_data
python scripts/orchestration/run_regime_hmm.py
```

### Error: `FileNotFoundError: configs/base.yaml`

**Solución**: Asegúrate de estar en `caria_data/` o usa path absoluto:
```powershell
python scripts/orchestration/run_regime_hmm.py --config C:\key\wise_adviser_cursor_context\notebooks\caria_data\configs\base.yaml
```

