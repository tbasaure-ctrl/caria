# Setup para Google Colab - Diagnóstico de Overfitting

Este documento explica cómo usar el notebook `diagnostic_overfitting_colab.ipynb` en Google Colab.

## Archivos Necesarios

Sube estos archivos a Google Drive en la estructura indicada:

```
caria_data/
├── data/
│   └── gold/
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
├── models/
│   ├── quality_model.pkl
│   ├── valuation_model.pkl
│   ├── momentum_model.pkl
│   └── feature_config.pkl
└── artifacts/
    └── models/
        └── epoch=20-val_loss=0.0038.ckpt  (opcional)
```

## Pasos para Ejecutar

1. **Subir archivos a Google Drive**
   - Crea una carpeta llamada `caria_data` en tu Google Drive
   - Sube los archivos manteniendo la estructura de carpetas mostrada arriba

2. **Abrir el notebook en Colab**
   - Sube el archivo `diagnostic_overfitting_colab.ipynb` a Google Colab
   - O abre directamente desde Drive si lo subiste ahí

3. **Ajustar la ruta de Drive**
   - En la celda de "Montar Google Drive", ajusta `DRIVE_BASE_PATH` si tu carpeta está en otra ubicación
   - Por defecto busca en: `/content/drive/MyDrive/caria_data`

4. **Ejecutar todas las celdas**
   - Ejecuta las celdas en orden (Runtime > Run All)
   - El notebook instalará dependencias automáticamente
   - Copiará los archivos desde Drive al workspace de Colab
   - Ejecutará el diagnóstico
   - Guardará resultados y los copiará de vuelta a Drive

## Estructura Esperada en Drive

Si prefieres otra estructura, ajusta las rutas en la celda "Configurar Rutas y Crear Estructura":

```python
# Ejemplo de rutas personalizadas:
src = drive_data_path / 'mi_carpeta' / 'datos' / f'{split}.parquet'
```

## Resultados

Los resultados se guardarán en:
- **JSON**: `caria_data/artifacts/diagnostics/overfitting_report_YYYYMMDD_HHMMSS.json`
- **Gráfico**: `caria_data/artifacts/diagnostics/overfitting_plot_YYYYMMDD_HHMMSS.png`

## Notas

- El notebook lee solo las columnas necesarias para ahorrar memoria
- Si tienes problemas de memoria, considera usar una GPU en Colab (Runtime > Change runtime type > GPU)
- Los modelos `.pkl` se cargan para detectar automáticamente qué features necesitan
- El checkpoint de PyTorch es opcional - solo se necesita si quieres evaluar SimpleFusionModel

## Troubleshooting

**Error: "Ruta no encontrada"**
- Verifica que `DRIVE_BASE_PATH` apunte a la ubicación correcta
- Asegúrate de que la carpeta `caria_data` existe en Drive

**Error: "No hay features disponibles"**
- Verifica que `feature_config.pkl` existe y está bien formado
- O que los modelos `.pkl` tienen `feature_names_in_` o `feature_names` configurados

**Error de memoria**
- Usa Runtime > Change runtime type > High-RAM
- O reduce el tamaño de los datasets (muestreo)




