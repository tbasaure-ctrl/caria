# Guía de Diagnóstico: Interpretación de Resultados CARIA

## Problemas Comunes y Soluciones

### 1. Sincronización NO Significativa (p ≥ 0.01)

**Síntoma:**
```
Sync Significativa: False (p=1.0000)
PLV observado: 0.15
PLV surrogate mean: 0.16 ± 0.02
```

**Interpretación:**
- La sincronización detectada NO es significativamente diferente del ruido
- Probablemente es un **artefacto del método** (filtrado, wavelets, etc.)
- **NO usar esta señal para trading**

**Soluciones:**

#### Opción A: Usar Correlation-Based Sync (Más Robusto)
```python
# El código ahora calcula también correlation-based sync
sync_corr = metrics['synchronization_corr']  # Usar esta en lugar de PLV
```

#### Opción B: Aumentar Rigor de Validación
```python
# Aumentar número de surrogates
sync_result = calculate_plv_sync(
    prices,
    n_surrogates=200  # En lugar de 50
)
```

#### Opción C: Verificar Wavelets
```python
# Verificar que wavelets funcionen correctamente
try:
    band_signals = wavelet_decompose_morlet(data, bands)
    print("✅ Wavelets funcionando")
except Exception as e:
    print(f"❌ Error en wavelets: {e}")
    # Usar bandpass como fallback
```

### 2. Cuadrantes Casi Vacíos (Muy Pocos Puntos)

**Síntoma:**
```
Q1: 0.00% (0/1)
Q2: 0.00% (0/0)  ← PROBLEMA: 0 puntos
Q3: 0.00% (0/1)
Q4: 0.00% (0/0)
```

**Causas Posibles:**

1. **Sincronización mayormente NaN**
   - **Solución**: El código ahora guarda PLV incluso si no es significativo
   - Verificar que `sync_rolling` tenga suficientes valores válidos

2. **Umbrales mal calibrados**
   - Si todos los puntos caen en Q1 o Q4, los umbrales están mal
   - **Solución**: Usar percentiles en lugar de medianas
   ```python
   entropy_threshold = entropy_aligned.quantile(0.5)  # Mediana (actual)
   # O usar terciles:
   entropy_threshold = entropy_aligned.quantile(0.33)  # Más balanceado
   ```

3. **Datos insuficientes**
   - Si hay muy pocos días con datos válidos
   - **Solución**: Verificar rango de fechas y completitud de datos

**Diagnóstico:**
```python
# Verificar distribución de cuadrantes
print(f"Distribución de cuadrantes:")
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    n_q = (quadrants == q).sum()
    print(f"  {q}: {n_q} puntos")
```

### 3. PLV Muy Bajo (< 0.2)

**Síntoma:**
```
Sync PLV: 0.1522 ± 0.0061
```

**Interpretación:**
- Para N=5 bandas, ruido blanco debería tener PLV ≈ 1/√5 ≈ 0.45
- PLV = 0.15 es **sospechosamente bajo**
- Puede indicar:
  1. **Problema en cálculo**: Wavelets o fase extraction fallando
  2. **Datos muy desincronizados**: Realmente no hay sincronización
  3. **Método inadecuado**: PLV no es apropiado para estos datos

**Soluciones:**

#### Verificar Cálculo de Fase
```python
# Verificar que las fases sean válidas
for name, phase in band_phases.items():
    print(f"{name}: fase range [{phase.min():.2f}, {phase.max():.2f}] rad")
    print(f"  NaN count: {np.isnan(phase).sum()}")
```

#### Usar Correlation-Based Sync
```python
# Más robusto y menos sensible a artefactos
corr_sync = correlation_based_sync(prices, bands)
print(f"Correlation sync: {corr_sync:.4f}")
```

### 4. Crisis Labels No Alineadas

**Síntoma:**
```
Total días de crisis marcados: 0
```

**Causa:**
- Las fechas de crisis no coinciden con las fechas en los datos
- Puede ser problema de formato de fecha o zona horaria

**Solución:**
```python
# Verificar fechas disponibles
print(f"Rango de datos: {df['date'].min()} a {df['date'].max()}")

# Buscar fechas cercanas a crisis
for crisis_date_str, crisis_name in KNOWN_CRISES.items():
    crisis_date = pd.to_datetime(crisis_date_str)
    date_diff = (df['date'] - crisis_date).abs()
    closest_idx = date_diff.idxmin()
    closest_date = df.loc[closest_idx, 'date']
    days_diff = abs((closest_date - crisis_date).days)
    print(f"{crisis_name}: fecha más cercana {closest_date.date()} (diferencia: {days_diff} días)")
```

### 5. Super-Criticality No Validada

**Síntoma:**
```
Q2 (Super-Criticality) NORM: 0.00%
⚠️ HIPÓTESIS NO VALIDADA
```

**Posibles Interpretaciones:**

#### A. Hipótesis Incorrecta
- Super-Criticality puede no ser el estado peligroso
- Quizás es Q3 (Low Entropy + High Sync) el verdadero peligro
- **Acción**: Analizar todos los cuadrantes, no solo Q2

#### B. Método Inadecuado
- Entropía normalizada puede estar eliminando la señal
- **Acción**: Comparar RAW vs NORMALIZADO cuidadosamente

#### C. Datos Insuficientes
- Muy pocos puntos en Q2 para calcular probabilidad confiable
- **Acción**: Usar bootstrap confidence intervals

**Solución: Bootstrap CI**
```python
from scipy import stats

def bootstrap_quadrant_probability(entropy, sync, crisis_labels, quadrant_mask, n_bootstrap=1000):
    """Bootstrap CI para probabilidad de crisis en cuadrante."""
    n_q = quadrant_mask.sum()
    n_crisis_q = (crisis_labels[quadrant_mask] == 1).sum()
    
    if n_q < 10:
        return {'mean': n_crisis_q / n_q if n_q > 0 else 0, 'ci_lower': 0, 'ci_upper': 0}
    
    # Bootstrap
    bootstrap_probs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(quadrant_mask.sum(), size=n_q, replace=True)
        crisis_sample = crisis_labels[quadrant_mask].iloc[indices]
        prob = (crisis_sample == 1).sum() / len(crisis_sample)
        bootstrap_probs.append(prob)
    
    bootstrap_probs = np.array(bootstrap_probs)
    return {
        'mean': np.mean(bootstrap_probs),
        'ci_lower': np.percentile(bootstrap_probs, 2.5),
        'ci_upper': np.percentile(bootstrap_probs, 97.5)
    }
```

## Checklist de Diagnóstico

Antes de interpretar resultados, verificar:

- [ ] **Sincronización significativa**: p < 0.01
- [ ] **PLV razonable**: Entre 0.2 y 0.8 (no demasiado bajo ni alto)
- [ ] **Distribución balanceada**: Cada cuadrante tiene >50 puntos
- [ ] **Crisis labels alineadas**: Al menos algunas crisis están marcadas
- [ ] **Datos completos**: <10% NaN en series principales
- [ ] **Rango de fechas adecuado**: Cubre períodos de crisis conocidas

## Interpretación Correcta de Resultados

### Resultados Válidos:
```
Sync Significativa: True (p=0.003)
PLV: 0.45 ± 0.05
Q2 (Super-Criticality): 25.3% [18.2%, 32.1%] (bootstrap CI)
```

### Resultados Inválidos (Actuales):
```
Sync Significativa: False (p=1.0000)
PLV: 0.15 ± 0.01
Q2 (Super-Criticality): 0.00% (0/0 puntos)
```

**Conclusión**: Los resultados actuales **NO son válidos** para validar Super-Criticality porque:
1. Sincronización no es significativa (artefacto)
2. Muy pocos puntos en Q2 (no hay suficientes datos)
3. PLV muy bajo (posible error en cálculo)

## Próximos Pasos Recomendados

1. **Usar Correlation-Based Sync** en lugar de PLV
2. **Aumentar n_surrogates** a 200+ para validación más estricta
3. **Verificar wavelets** funcionan correctamente
4. **Usar percentiles** en lugar de medianas para umbrales
5. **Bootstrap CI** para probabilidades de cuadrantes
6. **Comparar con método alternativo** (correlation sync)

## Código de Diagnóstico Rápido

```python
# Ejecutar después de calcular métricas
for name, metrics in caria_metrics.items():
    print(f"\n🔍 Diagnóstico {name}:")
    
    # 1. Verificar sincronización
    sync_val = metrics['sync_validation']
    if not sync_val['is_significant']:
        print(f"  ❌ Sync NO significativa (p={sync_val['p_value']:.4f})")
        print(f"     Usar correlation-based sync en su lugar")
    else:
        print(f"  ✅ Sync significativa")
    
    # 2. Verificar distribución
    sync = metrics['synchronization'].dropna()
    entropy = metrics['entropy_norm'].dropna()
    
    if len(sync) < len(metrics['dates']) * 0.9:
        print(f"  ⚠️ Muchos NaN en sync: {len(sync)}/{len(metrics['dates'])}")
    
    # 3. Verificar umbrales
    entropy_threshold = entropy.median()
    sync_threshold = sync.median()
    
    print(f"  Umbrales: Entropía={entropy_threshold:.4f}, Sync={sync_threshold:.4f}")
    
    # 4. Distribución esperada
    q1 = ((entropy >= entropy_threshold) & (sync < sync_threshold)).sum()
    q2 = ((entropy >= entropy_threshold) & (sync >= sync_threshold)).sum()
    q3 = ((entropy < entropy_threshold) & (sync >= sync_threshold)).sum()
    q4 = ((entropy < entropy_threshold) & (sync < sync_threshold)).sum()
    
    print(f"  Distribución: Q1={q1}, Q2={q2}, Q3={q3}, Q4={q4}")
    
    if q2 < 50:
        print(f"  ⚠️ Q2 tiene muy pocos puntos ({q2}), resultados no confiables")
```

---

**Última actualización**: Diciembre 2025  
**Versión**: 2.0 (Post-Diagnostic)













