# CARIA Real Data Validation - Google Colab

Este notebook valida el framework CARIA con datos reales usando la API de Financial Modeling Prep (FMP).

## 🚀 Inicio Rápido

### Opción 1: Usar el Script Python Directo

1. **Abre Google Colab**: https://colab.research.google.com/

2. **Sube el archivo** `CARIA_Real_Data_Validation.py` a Colab

3. **Ejecuta las celdas en orden** (divide el script en celdas según los comentarios `# CELDA X`)

### Opción 2: Crear Notebook desde Cero

1. **Crea un nuevo notebook** en Colab

2. **Copia y pega cada sección** del script `CARIA_Real_Data_Validation.py` en celdas separadas

3. **Ejecuta en orden**

## 📋 Requisitos Previos

### 1. API Key de Financial Modeling Prep

1. Regístrate gratis en: https://site.financialmodelingprep.com/developer/docs/
2. Obtén tu API key (gratis hasta cierto límite de requests)
3. Reemplaza `FMP_API_KEY = "TU_API_KEY_AQUI"` en el código

### 2. Instalación de Dependencias

El script instala automáticamente:
- PyWavelets (para wavelets Morlet)
- pandas, numpy, scipy
- matplotlib, seaborn
- requests

## 📊 Estructura del Notebook

### Celdas Principales:

1. **Instalación de Dependencias**
   ```python
   !pip install -q PyWavelets pandas numpy scipy scikit-learn requests matplotlib seaborn
   ```

2. **Configuración**
   - API Key de FMP
   - Símbolos a analizar (S&P 500, VIX, TLT, GLD)
   - Período de análisis (2010-2025)

3. **Descarga de Datos**
   - Descarga histórica desde FMP API
   - Procesamiento de retornos y volatilidad

4. **Funciones CARIA**
   - Entropía con normalización de volatilidad
   - Sincronización con wavelets y PLV
   - Surrogate testing (p < 0.01)

5. **Cálculo de Métricas**
   - Entropía RAW vs NORMALIZADA
   - Sincronización PLV con validación
   - Rolling metrics

6. **Validación Super-Criticality**
   - Mapeo a cuadrantes (Q1-Q4)
   - Probabilidades de crisis por cuadrante
   - Comparación RAW vs NORMALIZADO

7. **Visualizaciones**
   - Series temporales de métricas
   - Espacio de fase (Entropía vs Sync)
   - Marcadores de crisis conocidas

8. **Resumen y Conclusiones**
   - Estadísticas por activo
   - Validación de hipótesis
   - Próximos pasos

## 🔍 Validaciones Implementadas

### 1. Surrogate Data Testing (CRÍTICO)
- Genera 100 surrogates (datos barajados)
- Valida que sincronización sea real (p < 0.01)
- Rechaza señales si son artefactos

### 2. Volatility-Normalized Entropy
- Normaliza retornos: `z_t = (r_t - μ_t) / σ_t`
- Elimina efecto confusor de amplitud de volatilidad
- Compara RAW vs NORMALIZADO

### 3. Wavelet Decomposition
- Usa Morlet CWT (preferido sobre bandpass)
- Menos artefactos de fase
- Fallback a bandpass si PyWavelets falla

### 4. Phase-Locking Value (PLV)
- Más robusto que Kuramoto
- Validación estricta con surrogates
- p < 0.01 threshold

## 📈 Resultados Esperados

### Super-Criticality Validation:

**Si la hipótesis es correcta:**
- Q2 (High Entropy + High Sync) debería mostrar **>20% probabilidad de crisis**
- Esto debería persistir con entropía NORMALIZADA (no solo RAW)

**Si es un artefacto:**
- Q2 con entropía RAW muestra alta probabilidad
- Q2 con entropía NORMALIZADA muestra probabilidad baja
- → La hipótesis sería inválida (confundida por volatilidad)

### Sincronización:

**Validación exitosa:**
- PLV significativo (p < 0.01)
- Surrogate mean < observed PLV
- Z-score > 2.33 (para p < 0.01)

**Si falla validación:**
- p ≥ 0.01 → Rechazar señal (probable artefacto)
- Usar método alternativo o ajustar parámetros

## 🐛 Troubleshooting

### Error: "No se encontraron datos para [símbolo]"
- Verifica que el símbolo sea correcto en FMP
- Algunos símbolos pueden requerir formato diferente (ej: `SPY` en lugar de `^GSPC`)
- Verifica que tu API key tenga acceso a datos históricos

### Error: "PyWavelets not installed"
```python
!pip install PyWavelets
```

### Error: "Surrogate test failed"
- Reduce `n_surrogates` temporalmente (ej: 20 en lugar de 100)
- Verifica que los datos tengan suficiente longitud (>500 puntos)

### Cálculo muy lento
- Reduce `n_surrogates` en rolling sync (ej: 10-20)
- Usa ventanas más pequeñas para rolling (ej: 126 en lugar de 252)
- Considera usar solo un activo para pruebas iniciales

## 📝 Notas Importantes

1. **API Rate Limits**: FMP tiene límites de requests. Espera 0.5s entre requests.

2. **Tiempo de Ejecución**: 
   - Descarga de datos: ~30 segundos
   - Cálculo de métricas: ~5-10 minutos por activo
   - Rolling sync: ~10-20 minutos por activo (depende de n_surrogates)

3. **Memoria**: El notebook puede usar ~2-4 GB de RAM con todos los activos.

4. **Resultados**: Los gráficos se guardan automáticamente como PNG.

## 🔬 Próximos Pasos Después de Validación

1. **Bootstrap Confidence Intervals**: Calcular CI para probabilidades de cuadrantes
2. **Comparar Métodos**: Entropía Permutation vs Shannon vs Spectral
3. **Transfer Entropy**: Detectar dirección de acoplamiento
4. **Network Synchronization**: Sincronización cross-asset
5. **Backtesting**: Implementar estrategia con position sizing

## 📚 Referencias

- **FMP API Docs**: https://site.financialmodelingprep.com/developer/docs/
- **PyWavelets**: https://pywavelets.readthedocs.io/
- **CARIA Technical Report**: Ver `CARIA_Technical_Report_Detailed.md`

## ⚠️ Disclaimer

Este notebook es para **investigación y validación**. No es un sistema de trading listo para producción. Siempre valida con surrogate testing antes de usar señales en trading real.

---

**Autor**: CARIA Research Core  
**Fecha**: Diciembre 2025  
**Versión**: 2.0 (Post-Critical Review)













