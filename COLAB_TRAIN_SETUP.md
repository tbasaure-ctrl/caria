# Instrucciones para Entrenar Modelos Mejorados en Google Colab

## Archivos a Subir a Google Drive

Crea una carpeta en tu Google Drive llamada `caria_data` y sube los siguientes archivos:

### Estructura de Carpetas en Drive:

```
/MyDrive/caria_data/
├── data/
│   ├── gold/
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── test.parquet
│   └── silver/
│       └── macro/  (opcional, se creará automáticamente si no existe)
└── models/  (se creará automáticamente)
```

**Nota**: El notebook usa por defecto la ruta `/content/drive/MyDrive/caria_data`. Si prefieres otro nombre, ajusta `DRIVE_BASE_PATH` en la celda correspondiente.

### Archivos Requeridos:

1. **Datos Gold** (obligatorio):
   - Sube estos archivos a: `/MyDrive/caria_data/data/gold/`
   - `train.parquet`
   - `val.parquet`
   - `test.parquet`

2. **API Key de FRED** (obligatorio):
   - Necesitas obtener una API key gratuita de: https://fred.stlouisfed.org/docs/api/api_key.html
   - Se pedirá en el notebook

### Archivos Opcionales:

- Si ya tienes datos macro procesados, puedes subirlos a `/MyDrive/caria_data/data/silver/macro/macro_features.parquet`
- Si no, el notebook los descargará automáticamente desde FRED y los guardará ahí

## Pasos para Ejecutar

1. **Abrir el notebook en Colab**:
   - Sube `train_improved_models_colab.ipynb` a Google Colab
   - O abre directamente desde Drive si lo guardaste ahí

2. **Ejecutar celdas en orden**:
   - El notebook está diseñado para ejecutarse secuencialmente
   - Cada celda tiene instrucciones claras

3. **Configurar API Key de FRED**:
   - En la celda correspondiente, ingresa tu API key cuando se solicite

4. **Montar Google Drive**:
   - Se ejecutará automáticamente y pedirá permisos

5. **Esperar a que termine**:
   - El entrenamiento puede tardar varias horas dependiendo del tamaño de los datos
   - Los modelos se guardarán automáticamente en Drive

## Modelos que se Entrenarán

1. **Quality Model** (`improved_quality_model.pkl`):
   - Identifica empresas de alta calidad usando percentiles por fecha
   - Evita leakage temporal usando solo datos históricos
   - Features: ROIC, ROE, márgenes, crecimiento, contexto macro

2. **Valuation Model** (`improved_valuation_model.pkl`):
   - Predice retornos futuros usando DCF y múltiplos
   - Incorpora contexto macro (yield curve, credit spreads, commodities)
   - Features: Percentiles históricos de múltiplos, ratios fundamentales, indicadores macro

3. **Momentum Model** (`improved_momentum_model.pkl`):
   - Clasifica dirección de retornos usando indicadores técnicos
   - Prioriza volumen, SMAs (200/50), y RSI
   - Features: Volumen, SMAs, EMAs, RSI, MACD, posición relativa vs SMAs

## Outputs Generados

Todos los archivos se guardarán en:
- `/MyDrive/caria_data/models/`

Archivos generados:
- `improved_quality_model.pkl`
- `improved_valuation_model.pkl`
- `improved_momentum_model.pkl`
- `improved_feature_config.pkl` (configuración de features usadas)

## Troubleshooting

### Error: "No se encontraron datos en Drive"
- Verifica que la ruta en `DRIVE_BASE_PATH` sea correcta (por defecto: `/content/drive/MyDrive/caria_data`)
- Asegúrate de que los archivos `.parquet` estén en `/MyDrive/caria_data/data/gold/`
- Si usas otro nombre de carpeta, ajusta `DRIVE_BASE_PATH` en la celda 4 del notebook

### Error: "FRED API Key inválida"
- Verifica que la API key sea correcta
- Asegúrate de que no tenga espacios extra al inicio/final

### Error: "Memoria insuficiente"
- Reduce el tamaño del batch o usa una instancia con más RAM
- En Colab Pro puedes usar instancias con más memoria

### Error: "Features no encontradas"
- Algunas features pueden no estar disponibles en tus datos
- El notebook maneja esto automáticamente filtrando solo las features disponibles

## Notas Importantes

- Los modelos usan regularización fuerte para evitar overfitting
- El Quality Model usa percentiles por fecha para comparar empresas en el mismo contexto temporal
- El Valuation Model predice retornos futuros (5 años) en lugar de retornos inmediatos
- El Momentum Model es un clasificador binario (dirección positiva/negativa)

## Próximos Pasos

Después de entrenar los modelos:
1. Descarga los modelos desde Drive
2. Úsalos para inferencia en tu pipeline local
3. Evalúa el rendimiento con el script `check_overfitting.py`
4. Ajusta hiperparámetros si es necesario

