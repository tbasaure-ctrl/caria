"""
CARIA Real Data Validation - Google Colab Script
==================================================

Este script valida el framework CARIA con datos reales de Financial Modeling Prep API.

Para usar en Colab:
1. Sube este archivo a Colab
2. Ejecuta las celdas en orden
3. Reemplaza FMP_API_KEY con tu API key

Autor: CARIA Research Core
Fecha: Diciembre 2025
"""

# ==============================================================================
# CELDA 1: Instalación de Dependencias
# ==============================================================================
"""
# Ejecutar en Colab:
!pip install -q PyWavelets pandas numpy scipy scikit-learn requests matplotlib seaborn plotly
"""

# ==============================================================================
# CELDA 2: Imports y Configuración
# ==============================================================================
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
from scipy import signal, stats
from scipy.signal import hilbert, butter, filtfilt
import pywt
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
np.random.seed(42)

# Configuración FMP API
FMP_API_KEY = "TU_API_KEY_AQUI"  # ⚠️ CAMBIAR
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# Símbolos
SYMBOLS = {
    'SP500': '^GSPC',
    'VIX': '^VIX',
    'TLT': 'TLT',
    'GLD': 'GLD'
}

START_DATE = "2010-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Configuración CARIA
DEFAULT_BANDS = {
    'ultra_fast': (1, 5),
    'short': (5, 20),
    'medium': (20, 60),
    'long': (60, 252),
    'ultra_long': (252, 504)
}

PHYSICS_WEIGHTS = {
    'ultra_fast': 0.05,
    'short': 0.10,
    'medium': 0.35,
    'long': 0.25,
    'ultra_long': 0.25
}

# ==============================================================================
# CELDA 3: Descarga de Datos FMP
# ==============================================================================
def download_fmp_data(symbol: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """Descarga datos históricos de FMP."""
    url = f"{FMP_BASE_URL}/historical-price-full/{symbol}"
    params = {
        'from': start_date,
        'to': end_date,
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'historical' not in data:
            print(f"⚠️ No se encontraron datos para {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        print(f"✅ {symbol}: {len(df)} días ({df['date'].min()} a {df['date'].max()})")
        return df
        
    except Exception as e:
        print(f"❌ Error descargando {symbol}: {e}")
        return pd.DataFrame()

# Descargar datos
data_dict = {}
for name, symbol in SYMBOLS.items():
    df = download_fmp_data(symbol, START_DATE, END_DATE, FMP_API_KEY)
    if not df.empty:
        data_dict[name] = df
    import time
    time.sleep(0.5)

# ==============================================================================
# CELDA 4: Funciones CARIA (Entropía Normalizada)
# ==============================================================================
def shannon_entropy_normalized(
    data: np.ndarray,
    bins: str = 'fd',
    normalize: bool = True,
    volatility_normalize: bool = True,
    rolling_window: int = 30
) -> float:
    """Shannon entropy con normalización de volatilidad (CRÍTICO)."""
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]
    
    if len(data) < 30:
        return np.nan
    
    # CRÍTICO: Normalización de volatilidad
    if volatility_normalize:
        data_series = pd.Series(data)
        rolling_mean = data_series.rolling(window=rolling_window, min_periods=min(rolling_window, len(data)//4)).mean()
        rolling_std = data_series.rolling(window=rolling_window, min_periods=min(rolling_window, len(data)//4)).std()
        
        rolling_mean = rolling_mean.fillna(method='bfill').fillna(data_series.mean())
        rolling_std = rolling_std.fillna(method='bfill').fillna(data_series.std())
        rolling_std = rolling_std.replace(0, np.nan).fillna(data_series.std())
        
        data = ((data_series - rolling_mean) / rolling_std).values
        data = data[~np.isnan(data)]
    
    # Calcular histograma
    counts, _ = np.histogram(data, bins=bins, density=False)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    
    # Entropía
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    # Normalizar
    if normalize:
        max_entropy = np.log2(len(counts))
        entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return entropy

def rolling_shannon_entropy_normalized(
    returns: pd.Series,
    window: int = 30,
    volatility_normalize: bool = True
) -> pd.Series:
    """Rolling Shannon entropy con normalización de volatilidad."""
    result = pd.Series(index=returns.index, dtype=float)
    
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window:i+1].values
        try:
            ent = shannon_entropy_normalized(
                window_data,
                volatility_normalize=volatility_normalize,
                rolling_window=window
            )
            result.iloc[i] = ent
        except:
            result.iloc[i] = np.nan
    
    return result

# ==============================================================================
# CELDA 5: Funciones CARIA (Sincronización con Wavelets y PLV)
# ==============================================================================
def extract_phase_detrended(signal_data: np.ndarray) -> np.ndarray:
    """Extrae fase instantánea con detrending."""
    signal_data = np.asarray(signal_data).flatten()
    
    # Detrend
    n = len(signal_data)
    x = np.arange(n)
    coeffs = np.polyfit(x, signal_data, 1)
    trend = np.polyval(coeffs, x)
    signal_data = signal_data - trend - np.mean(signal_data)
    
    # Hilbert transform
    analytic = hilbert(signal_data)
    phase = np.angle(analytic)
    phase = np.unwrap(phase)
    
    return phase

def wavelet_decompose_morlet(data: np.ndarray, bands: dict) -> dict:
    """Descomposición con wavelets Morlet (PREFERIDO)."""
    data = np.asarray(data).flatten()
    omega0 = 6.0
    dt = 1.0
    
    decomposed = {}
    
    for name, (low_period, high_period) in bands.items():
        low_scale = low_period / (2 * np.pi * omega0 * dt)
        high_scale = high_period / (2 * np.pi * omega0 * dt)
        
        n_scales = max(10, int((high_scale - low_scale) / 0.5))
        band_scales = np.linspace(low_scale, high_scale, n_scales)
        
        try:
            coefficients, _ = pywt.cwt(data, band_scales, 'cmor', dt=dt)
            band_signal = np.real(coefficients).mean(axis=0)
            decomposed[name] = band_signal
        except:
            # Fallback a bandpass
            low_freq = 1.0 / high_period
            high_freq = 1.0 / low_period
            nyq = 0.5
            low = max(0.001, min(low_freq / nyq, 0.999))
            high = max(low + 0.001, min(high_freq / nyq, 0.999))
            b, a = butter(4, [low, high], btype='band')
            decomposed[name] = filtfilt(b, a, data)
    
    return decomposed

def phase_locking_value(phases1: np.ndarray, phases2: np.ndarray) -> float:
    """Phase-Locking Value entre dos series de fase."""
    phase_diff = phases1 - phases2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return plv

def correlation_based_sync(
    data: np.ndarray,
    bands: dict = None,
    window: int = 30
) -> float:
    """
    Sincronización basada en correlación entre bandas (MÁS ROBUSTA).
    
    Alternativa a PLV cuando surrogate testing falla.
    Calcula correlación promedio entre todas las parejas de bandas.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    
    # Descomponer
    try:
        band_signals = wavelet_decompose_morlet(data, bands)
    except:
        # Fallback a bandpass
        band_signals = {}
        for name, (low_period, high_period) in bands.items():
            low_freq = 1.0 / high_period
            high_freq = 1.0 / low_period
            nyq = 0.5
            low = max(0.001, min(low_freq / nyq, 0.999))
            high = max(low + 0.001, min(high_freq / nyq, 0.999))
            b, a = butter(4, [low, high], btype='band')
            band_signals[name] = filtfilt(b, a, data)
    
    # Calcular correlaciones entre todas las parejas
    band_names = list(band_signals.keys())
    n_bands = len(band_names)
    correlations = []
    
    for i, name1 in enumerate(band_names):
        for j, name2 in enumerate(band_names):
            if i < j:  # Solo pares únicos
                corr = np.corrcoef(band_signals[name1], band_signals[name2])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))  # Valor absoluto
    
    # Media de correlaciones
    mean_corr = np.mean(correlations) if len(correlations) > 0 else 0.0
    
    return mean_corr

def calculate_plv_sync(
    data: np.ndarray,
    bands: dict = None,
    method: str = 'wavelet',
    n_surrogates: int = 100
) -> dict:
    """Calcula sincronización usando PLV con validación estricta (p < 0.01)."""
    if bands is None:
        bands = DEFAULT_BANDS
    
    # Descomponer
    if method == 'wavelet':
        band_signals = wavelet_decompose_morlet(data, bands)
    else:
        # Fallback a bandpass
        band_signals = {}
        for name, (low_period, high_period) in bands.items():
            low_freq = 1.0 / high_period
            high_freq = 1.0 / low_period
            nyq = 0.5
            low = max(0.001, min(low_freq / nyq, 0.999))
            high = max(low + 0.001, min(high_freq / nyq, 0.999))
            b, a = butter(4, [low, high], btype='band')
            band_signals[name] = filtfilt(b, a, data)
    
    # Extraer fases
    band_phases = {}
    for name, band_data in band_signals.items():
        band_phases[name] = extract_phase_detrended(band_data)
    
    # Calcular PLV entre todos los pares
    band_names = list(band_phases.keys())
    n_bands = len(band_names)
    plv_matrix = np.zeros((n_bands, n_bands))
    
    for i, name1 in enumerate(band_names):
        for j, name2 in enumerate(band_names):
            if i <= j:
                plv = phase_locking_value(band_phases[name1], band_phases[name2])
                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv
    
    # Mean PLV (excluyendo diagonal)
    mask = ~np.eye(n_bands, dtype=bool)
    mean_plv = plv_matrix[mask].mean()
    max_plv = plv_matrix[mask].max()
    
    # Surrogate validation (STRICTO: p < 0.01)
    surrogates = [np.random.permutation(data) for _ in range(n_surrogates)]
    surrogate_plvs = []
    
    for surrogate in surrogates:
        try:
            if method == 'wavelet':
                surr_bands = wavelet_decompose_morlet(surrogate, bands)
            else:
                surr_bands = {}
                for name, (low_period, high_period) in bands.items():
                    low_freq = 1.0 / high_period
                    high_freq = 1.0 / low_period
                    nyq = 0.5
                    low = max(0.001, min(low_freq / nyq, 0.999))
                    high = max(low + 0.001, min(high_freq / nyq, 0.999))
                    b, a = butter(4, [low, high], btype='band')
                    surr_bands[name] = filtfilt(b, a, surrogate)
            
            surr_phases = {name: extract_phase_detrended(band_data) for name, band_data in surr_bands.items()}
            
            surr_plv_matrix = np.zeros((n_bands, n_bands))
            for i, name1 in enumerate(band_names):
                for j, name2 in enumerate(band_names):
                    if i <= j:
                        plv = phase_locking_value(surr_phases[name1], surr_phases[name2])
                        surr_plv_matrix[i, j] = plv
                        surr_plv_matrix[j, i] = plv
            
            surr_mask = ~np.eye(n_bands, dtype=bool)
            surr_mean_plv = surr_plv_matrix[surr_mask].mean()
            surrogate_plvs.append(surr_mean_plv)
        except:
            continue
    
    surrogate_plvs = np.array(surrogate_plvs)
    
    if len(surrogate_plvs) > 0:
        surrogate_mean = np.mean(surrogate_plvs)
        surrogate_std = np.std(surrogate_plvs)
        
        if surrogate_std > 0:
            z_score = (mean_plv - surrogate_mean) / surrogate_std
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            z_score = np.inf if mean_plv > surrogate_mean else -np.inf
            p_value = 0.0 if mean_plv > surrogate_mean else 1.0
        
        is_significant = p_value < 0.01  # ESTRICTO
    else:
        surrogate_mean = np.nan
        surrogate_std = np.nan
        z_score = np.nan
        p_value = np.nan
        is_significant = False
    
    return {
        'mean_plv': mean_plv,
        'max_plv': max_plv,
        'plv_matrix': plv_matrix,
        'band_names': band_names,
        'surrogate_mean': surrogate_mean,
        'surrogate_std': surrogate_std,
        'p_value': p_value,
        'is_significant': is_significant,
        'z_score': z_score
    }

# ==============================================================================
# CELDA 6: Procesamiento de Datos
# ==============================================================================
processed_data = {}
for name, df in data_dict.items():
    df = df.copy()
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(window=30).std() * np.sqrt(252)
    df = df.dropna()
    processed_data[name] = df
    print(f"✅ {name}: {len(df)} días procesados")

# ==============================================================================
# CELDA 7: Cálculo de Métricas CARIA
# ==============================================================================
caria_metrics = {}

for name, df in processed_data.items():
    print(f"\n📈 Procesando {name}...")
    
    returns = df['returns'].values
    prices = df['close'].values
    
    # Entropía RAW y NORMALIZADA
    entropy_raw = rolling_shannon_entropy_normalized(
        pd.Series(returns, index=df.index),
        volatility_normalize=False
    )
    
    entropy_norm = rolling_shannon_entropy_normalized(
        pd.Series(returns, index=df.index),
        volatility_normalize=True
    )
    
    # Sincronización con PLV (usar muestra representativa si datos muy largos)
    print(f"  Calculando sincronización...")
    
    # Para datos muy largos, usar muestra de 1000 puntos para velocidad
    if len(prices) > 2000:
        print(f"    Datos muy largos ({len(prices)} puntos), usando muestra de 1000 puntos")
        sample_indices = np.linspace(0, len(prices)-1, 1000, dtype=int)
        prices_sample = prices[sample_indices]
    else:
        prices_sample = prices
    
    sync_result = calculate_plv_sync(
        prices_sample,
        method='wavelet',
        n_surrogates=50
    )
    
    # Diagnóstico de sincronización
    print(f"    PLV observado: {sync_result['mean_plv']:.4f}")
    print(f"    PLV surrogate mean: {sync_result['surrogate_mean']:.4f} ± {sync_result['surrogate_std']:.4f}")
    print(f"    p-value: {sync_result['p_value']:.4f}")
    print(f"    Significativo (p<0.01): {sync_result['is_significant']}")
    
    if not sync_result['is_significant']:
        print(f"    ⚠️ ADVERTENCIA: Sincronización NO significativa (probable artefacto)")
        print(f"       PLV observado ({sync_result['mean_plv']:.4f}) no es mayor que surrogate mean")
        print(f"       Considerar:")
        print(f"       - Aumentar n_surrogates")
        print(f"       - Verificar que wavelets funcionen correctamente")
        print(f"       - Usar método alternativo (correlation-based sync)")
    
    # Rolling synchronization
    # Rolling synchronization (OPTIMIZADO: calcular cada 10 días para velocidad)
    sync_rolling = pd.Series(index=df.index, dtype=float)
    window = 252  # 1 año
    step = 10  # Calcular cada 10 días (más rápido)
    
    print(f"  Calculando sincronización rolling (ventana={window}, step={step})...")
    print(f"    Esto puede tardar varios minutos...")
    
    n_calculations = 0
    n_significant = 0
    
    for i in range(window, len(prices), step):
        window_prices = prices[i-window:i+1]
        try:
            sync_window = calculate_plv_sync(
                window_prices,
                method='wavelet',
                n_surrogates=20  # Reducido para velocidad
            )
            
            # Guardar valor (significativo o no) para análisis
            sync_rolling.iloc[i] = sync_window['mean_plv']
            
            # Forward-fill para los días intermedios
            if i + step < len(prices):
                sync_rolling.iloc[i:i+step] = sync_window['mean_plv']
            
            n_calculations += 1
            if sync_window['is_significant']:
                n_significant += 1
                
        except Exception as e:
            sync_rolling.iloc[i] = np.nan
            if n_calculations < 5:  # Solo mostrar primeros errores
                print(f"    Error en índice {i}: {e}")
        
        if n_calculations % 10 == 0:
            print(f"    Progreso: {n_calculations} cálculos, {n_significant} significativos ({n_significant/max(n_calculations,1)*100:.1f}%)")
    
    # Forward-fill valores NaN restantes
    sync_rolling = sync_rolling.fillna(method='ffill').fillna(method='bfill')
    
    # Estadísticas finales
    valid_sync = sync_rolling.dropna()
    print(f"  ✅ Rolling sync completado:")
    print(f"     Cálculos: {n_calculations}, Significativos: {n_significant} ({n_significant/max(n_calculations,1)*100:.1f}%)")
    if len(valid_sync) > 0:
        print(f"     PLV promedio: {valid_sync.mean():.4f} ± {valid_sync.std():.4f}")
        print(f"     PLV min/max: {valid_sync.min():.4f} / {valid_sync.max():.4f}")
    print(f"     Puntos válidos: {len(valid_sync)}/{len(sync_rolling)}")
    
    if n_significant / max(n_calculations, 1) < 0.1:
        print(f"     ⚠️ ADVERTENCIA: Solo {n_significant/max(n_calculations,1)*100:.1f}% de ventanas muestran sincronización significativa")
        print(f"        Esto sugiere que la sincronización puede ser un artefacto del método")
        print(f"        Considerar usar correlation-based sync como alternativa")
    
    # Calcular también correlation-based sync como alternativa
    print(f"  Calculando correlation-based sync (alternativa más robusta)...")
    corr_sync_rolling = pd.Series(index=df.index, dtype=float)
    
    for i in range(window, len(prices), step):
        window_prices = prices[i-window:i+1]
        try:
            corr_sync = correlation_based_sync(window_prices, window=window)
            corr_sync_rolling.iloc[i] = corr_sync
            if i + step < len(prices):
                corr_sync_rolling.iloc[i:i+step] = corr_sync
        except:
            corr_sync_rolling.iloc[i] = np.nan
    
    corr_sync_rolling = corr_sync_rolling.fillna(method='ffill').fillna(method='bfill')
    
    if len(corr_sync_rolling.dropna()) > 0:
        print(f"     Correlation sync promedio: {corr_sync_rolling.mean():.4f} ± {corr_sync_rolling.std():.4f}")
    
    caria_metrics[name] = {
        'entropy_raw': entropy_raw,
        'entropy_norm': entropy_norm,
        'synchronization': sync_rolling,
        'synchronization_corr': corr_sync_rolling,  # Alternativa
        'volatility': df['volatility'],
        'returns': df['returns'],
        'prices': df['close'],
        'dates': df['date'],
        'sync_validation': sync_result
    }
    
    print(f"✅ {name} completado")
    print(f"   Entropía raw: {entropy_raw.mean():.4f} ± {entropy_raw.std():.4f}")
    print(f"   Entropía norm: {entropy_norm.mean():.4f} ± {entropy_norm.std():.4f}")
    print(f"   Sync PLV: {sync_rolling.mean():.4f} ± {sync_rolling.std():.4f}")
    print(f"   Sync significativa: {sync_result['is_significant']} (p={sync_result['p_value']:.4f})")

# ==============================================================================
# CELDA 8: Validación Super-Criticality
# ==============================================================================
KNOWN_CRISES = {
    '2010-05-06': 'Flash Crash',
    '2011-08-08': 'US Downgrade',
    '2015-08-24': 'China Crash',
    '2018-02-05': 'Volatility Spike',
    '2020-03-16': 'COVID Crash',
    '2022-03-07': 'Ukraine/Rate Hike',
    '2023-03-10': 'SVB Collapse'
}

def calculate_quadrant_probabilities(
    entropy: pd.Series,
    sync: pd.Series,
    dates: pd.Series,
    crisis_dates: dict,
    horizon: int = 5
) -> dict:
    """
    Calcula probabilidades de crisis por cuadrante.
    
    CRÍTICO: Usa fechas reales del DataFrame, no índices numéricos.
    """
    # Alinear series por índice
    common_idx = entropy.index.intersection(sync.index).intersection(dates.index)
    
    if len(common_idx) == 0:
        print("⚠️ ERROR: No hay índices comunes entre entropy, sync y dates")
        return {}, pd.Series(), pd.Series()
    
    entropy_aligned = entropy.loc[common_idx]
    sync_aligned = sync.loc[common_idx]
    dates_aligned = dates.loc[common_idx]
    
    # Filtrar NaN
    valid_mask = entropy_aligned.notna() & sync_aligned.notna()
    entropy_aligned = entropy_aligned[valid_mask]
    sync_aligned = sync_aligned[valid_mask]
    dates_aligned = dates_aligned[valid_mask]
    
    if len(entropy_aligned) < 100:
        print(f"⚠️ ADVERTENCIA: Solo {len(entropy_aligned)} puntos válidos después de filtrar NaN")
    
    # Umbrales (medianas)
    entropy_threshold = entropy_aligned.median()
    sync_threshold = sync_aligned.median()
    
    print(f"  Umbrales: Entropía={entropy_threshold:.4f}, Sync={sync_threshold:.4f}")
    
    # Crear crisis labels usando FECHAS REALES
    crisis_labels = pd.Series(0, index=entropy_aligned.index)
    
    crisis_count = 0
    for crisis_date_str, crisis_name in crisis_dates.items():
        try:
            crisis_date = pd.to_datetime(crisis_date_str)
            
            # Buscar fecha más cercana en los datos
            date_diff = (dates_aligned - crisis_date).abs()
            closest_idx = date_diff.idxmin()
            closest_date = dates_aligned.loc[closest_idx]
            
            # Si la fecha más cercana está dentro de 10 días, usar esa ventana
            if abs((closest_date - crisis_date).days) <= 10:
                # Marcar días dentro del horizonte ANTES de la crisis
                mask = (dates_aligned >= closest_date - pd.Timedelta(days=horizon)) & \
                       (dates_aligned < closest_date) & \
                       (dates_aligned.index.isin(entropy_aligned.index))
                
                n_marked = mask.sum()
                if n_marked > 0:
                    crisis_labels.loc[mask] = 1
                    crisis_count += n_marked
                    print(f"    {crisis_name} ({crisis_date_str}): {n_marked} días marcados (fecha más cercana: {closest_date.date()})")
        except Exception as e:
            print(f"    Error procesando {crisis_date_str}: {e}")
            continue
    
    print(f"  Total días de crisis marcados: {crisis_count}")
    
    # Clasificar en cuadrantes
    quadrants = pd.Series('', index=entropy_aligned.index)
    q1_mask = (entropy_aligned >= entropy_threshold) & (sync_aligned < sync_threshold)
    q2_mask = (entropy_aligned >= entropy_threshold) & (sync_aligned >= sync_threshold)
    q3_mask = (entropy_aligned < entropy_threshold) & (sync_aligned >= sync_threshold)
    q4_mask = (entropy_aligned < entropy_threshold) & (sync_aligned < sync_threshold)
    
    quadrants[q1_mask] = 'Q1'
    quadrants[q2_mask] = 'Q2'
    quadrants[q3_mask] = 'Q3'
    quadrants[q4_mask] = 'Q4'
    
    # Verificar que todos los puntos estén clasificados
    unclassified = (quadrants == '').sum()
    if unclassified > 0:
        print(f"  ⚠️ ADVERTENCIA: {unclassified} puntos no clasificados (puede haber NaN en sync)")
    
    # Diagnóstico de distribución de cuadrantes
    print(f"  Distribución de cuadrantes:")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        n_q = (quadrants == q).sum()
        pct_q = n_q / len(quadrants) * 100 if len(quadrants) > 0 else 0
        print(f"    {q}: {n_q} puntos ({pct_q:.1f}%)")
    
    # Estadísticas por cuadrante
    results = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        mask = quadrants == q
        n_total = mask.sum()
        n_crisis = (crisis_labels[mask] == 1).sum()
        prob = n_crisis / n_total if n_total > 0 else 0
        
        results[q] = {
            'probability': prob,
            'n_total': n_total,
            'n_crisis': n_crisis,
            'percentage_of_total': n_total / len(quadrants) if len(quadrants) > 0 else 0
        }
    
    # Diagnóstico
    total_points = len(quadrants)
    total_crisis = crisis_labels.sum()
    baseline_prob = total_crisis / total_points if total_points > 0 else 0
    
    print(f"  Puntos totales: {total_points}")
    print(f"  Probabilidad baseline (crisis/total): {baseline_prob*100:.2f}%")
    
    return results, quadrants, crisis_labels

# Validar Super-Criticality
super_criticality_results = {}

for name, metrics in caria_metrics.items():
    print(f"\n🔬 Validando Super-Criticality para {name}...")
    print(f"  Datos disponibles: {len(metrics['dates'])} días")
    print(f"  Rango de fechas: {metrics['dates'].min()} a {metrics['dates'].max()}")
    
    # CRÍTICO: Pasar fechas reales del DataFrame
    results_raw, quadrants_raw, crisis_labels = calculate_quadrant_probabilities(
        metrics['entropy_raw'],
        metrics['synchronization'],
        metrics['dates'],  # ← FECHAS REALES
        KNOWN_CRISES
    )
    
    results_norm, quadrants_norm, _ = calculate_quadrant_probabilities(
        metrics['entropy_norm'],
        metrics['synchronization'],
        metrics['dates'],  # ← FECHAS REALES
        KNOWN_CRISES
    )
    
    super_criticality_results[name] = {
        'raw': results_raw,
        'normalized': results_norm,
        'quadrants_raw': quadrants_raw,
        'quadrants_norm': quadrants_norm,
        'crisis_labels': crisis_labels
    }
    
    print(f"\n  📊 Resultados con Entropía RAW:")
    for q, res in results_raw.items():
        quadrant_names = {
            'Q1': 'Gas (High Entropy, Low Sync)',
            'Q2': 'Super-Criticality (High Entropy, High Sync)',
            'Q3': 'Solid (Low Entropy, High Sync)',
            'Q4': 'Liquid (Low Entropy, Low Sync)'
        }
        print(f"     {q} - {quadrant_names[q]}:")
        print(f"        Probabilidad: {res['probability']*100:.2f}%")
        print(f"        Puntos: {res['n_crisis']}/{res['n_total']} ({res['percentage_of_total']*100:.1f}% del total)")
    
    print(f"\n  📊 Resultados con Entropía NORMALIZADA:")
    for q, res in results_norm.items():
        quadrant_names = {
            'Q1': 'Gas (High Entropy, Low Sync)',
            'Q2': 'Super-Criticality (High Entropy, High Sync)',
            'Q3': 'Solid (Low Entropy, High Sync)',
            'Q4': 'Liquid (Low Entropy, Low Sync)'
        }
        print(f"     {q} - {quadrant_names[q]}:")
        print(f"        Probabilidad: {res['probability']*100:.2f}%")
        print(f"        Puntos: {res['n_crisis']}/{res['n_total']} ({res['percentage_of_total']*100:.1f}% del total)")
    
    q2_raw = results_raw['Q2']['probability']
    q2_norm = results_norm['Q2']['probability']
    
    print(f"\n  🔍 Q2 (Super-Criticality) - RAW: {q2_raw*100:.2f}%, NORM: {q2_norm*100:.2f}%")
    
    # Validación más robusta
    if results_norm['Q2']['n_total'] < 50:
        print(f"  ⚠️ MUY POCOS PUNTOS EN Q2: Solo {results_norm['Q2']['n_total']} puntos")
        print(f"     Esto hace que las probabilidades no sean confiables")
        print(f"     Posibles causas:")
        print(f"     1. Sincronización muy baja (mayoría de puntos en Q4)")
        print(f"     2. Entropía muy alta (mayoría de puntos en Q1)")
        print(f"     3. Umbrales mal calibrados")
    
    if q2_norm > 0.20:
        print(f"  ✅ HIPÓTESIS VALIDADA: Q2 muestra alta probabilidad ({q2_norm*100:.1f}%)")
    elif q2_norm > 0.10:
        print(f"  ⚠️ HIPÓTESIS PARCIALMENTE VALIDADA: Q2 muestra probabilidad moderada ({q2_norm*100:.1f}%)")
    else:
        print(f"  ❌ HIPÓTESIS NO VALIDADA: Q2 muestra probabilidad baja ({q2_norm*100:.1f}%)")
        print(f"     Posibles causas:")
        print(f"     - Muy pocos puntos en Q2 ({results_norm['Q2']['n_total']} puntos)")
        print(f"     - Sincronización no significativa (p={metrics['sync_validation']['p_value']:.4f})")
        print(f"     - Crisis labels no alineadas correctamente")
        print(f"     - Hipótesis puede ser incorrecta para este activo")
    
    if abs(q2_raw - q2_norm) > 0.05:
        print(f"  ⚠️ DIFERENCIA SIGNIFICATIVA RAW vs NORM: {abs(q2_raw-q2_norm)*100:.1f}pp")
        print(f"     → La normalización afecta significativamente los resultados")
    else:
        print(f"  ✅ Hipótesis robusta a normalización")

# ==============================================================================
# CELDA 9: Visualizaciones
# ==============================================================================
if 'SP500' in caria_metrics:
    metrics = caria_metrics['SP500']
    dates = metrics['dates']
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    axes[0].plot(dates, metrics['prices'], label='S&P 500', color='black', linewidth=1)
    axes[0].set_title('S&P 500 Prices', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(dates, metrics['entropy_raw'], label='Entropía RAW', alpha=0.7, color='blue')
    axes[1].plot(dates, metrics['entropy_norm'], label='Entropía NORMALIZADA', alpha=0.7, color='red')
    axes[1].set_title('Shannon Entropy (Raw vs Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Entropía Normalizada')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(dates, metrics['synchronization'], label='PLV Synchronization', color='green', alpha=0.7)
    axes[2].axhline(y=metrics['synchronization'].median(), color='red', linestyle='--', label='Mediana')
    axes[2].set_title('Phase-Locking Value (PLV) Synchronization', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('PLV')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(dates, metrics['volatility'], label='Volatilidad Realizada', color='orange', alpha=0.7)
    axes[3].set_title('Realized Volatility (30-day rolling)', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('Volatilidad Anualizada')
    axes[3].set_xlabel('Fecha')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('caria_sp500_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==============================================================================
# CELDA 10: Resumen Final
# ==============================================================================
print("=" * 80)
print("RESUMEN DE VALIDACIÓN CARIA CON DATOS REALES")
print("=" * 80)

for name, metrics in caria_metrics.items():
    print(f"\n📊 {name}:")
    print(f"   Entropía RAW: {metrics['entropy_raw'].mean():.4f} ± {metrics['entropy_raw'].std():.4f}")
    print(f"   Entropía NORM: {metrics['entropy_norm'].mean():.4f} ± {metrics['entropy_norm'].std():.4f}")
    print(f"   Sync PLV: {metrics['synchronization'].mean():.4f} ± {metrics['synchronization'].std():.4f}")
    
    sync_val = metrics['sync_validation']
    print(f"   Sync Significativa: {sync_val['is_significant']} (p={sync_val['p_value']:.4f})")
    
    if name in super_criticality_results:
        results = super_criticality_results[name]
        q2_norm = results['normalized']['Q2']['probability']
        q2_raw = results['raw']['Q2']['probability']
        
        print(f"   Q2 (Super-Criticality) RAW: {q2_raw*100:.2f}%")
        print(f"   Q2 (Super-Criticality) NORM: {q2_norm*100:.2f}%")
        
        if q2_norm > 0.20:
            print(f"   ✅ HIPÓTESIS VALIDADA: Q2 muestra alta probabilidad ({q2_norm*100:.1f}%)")
        else:
            print(f"   ⚠️ HIPÓTESIS NO VALIDADA: Q2 muestra probabilidad baja ({q2_norm*100:.1f}%)")

print("\n" + "=" * 80)
print("VALIDACIÓN COMPLETADA")
print("=" * 80)

