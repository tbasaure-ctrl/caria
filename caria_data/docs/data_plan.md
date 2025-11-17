# Data Plan — Caria Historical Dataset

## Objetivo
Establecer el diseño de datos para construir un dataset histórico multimodal (1900–presente) que combine señales macroeconómicas, commodities, FX, fundamentals corporativos, price action técnico y anotaciones de regímenes/psicología colectiva. El resultado debe alimentar los pipelines `bronze → silver → gold` y habilitar entrenamiento reproducible.

## Entidades Principales

| Entidad | Identificador | Frecuencia | Descripción |
|---------|---------------|------------|-------------|
| `macro_series` | `series_id` (FRED) | mensual / trimestral | Indicadores macro (CPI, PCE, GDP, Unemployment, Industrial Production, Yield Curve spreads) |
| `commodities` | `symbol` | diario | Precios spot/futuros (WTI, Brent, Gold, Silver, Copper, Wheat) |
| `fx_rates` | `pair` | diario | USD Index, EURUSD, GBPUSD, USDJPY, CNYUSD, Emerging Markets basket |
| `equities` | `ticker` | diario | Precios OHLCV para universos corporativos (Top performers + quebradas) |
| `fundamentals` | `ticker` + `fiscal_period` | trimestral/anual | ROIC, ROIIC, Revenue Growth, Net Debt, R&D, Capex, FCF Yield, Margins |
| `technical_indicators` | `ticker` | diario | SMA/EMA (20/50/200), RSI, MACD, Bollinger Bands, ATR |
| `events` | `event_id` | puntual | Regímenes etiquetados (recesión, depresión, burbuja, crisis), noticias relevantes, discursos |
| `sentiment_news` | `source_id` | diario | Sentimiento agregado por tema/sector para periodos críticos |

## Fuentes y API

- **FRED API**: CPIAUCSL, PCEPI, GDPC1, UNRATE, GS10, T10Y2Y, Industrial Production Index, M2SL, indicadores de recesión (USREC, USRECD).
- **EIA / Quandl**: Energía (WTI, Brent), metales (Gold, Silver), agrícolas (Wheat, Corn).
- **Stooq / AlphaVantage / Tiingo**: Series FX históricas cuando FRED no cubra pares específicos.
- **FMP API**: Fundamentals y precios OHLCV para universo de empresas (Top 50 outperformers + 50 quebradas). Incluye metadata de IPO, sector, país.
- **Stooq / Yahoo Finance**: Complemento para índices (Dow Jones, S&P 500, Nasdaq) y ETFs temáticos.
- **Noticias históricas**: New York Times Archive API, Global News Database (si disponible), bibliotecas públicas para eventos clave.
- **Listas de regímenes**: NBER (recesiones USA), BIS/IMF (crisis bancarias), dataset "Global Financial Data" para episodios de burbujas.

## Granularidad y Versionado

- `bronze/`: datos crudos según frecuencia original, particionados por `source/YYYY-MM-DD/`.
- `silver/`: normalizados a periodicidades estándar: diario (mercados), mensual (macro), trimestral (fundamentals). Todas las series se extienden a calendario `business day` usando forward-fill controlado (`max_gap` configurable).
- `gold/`: ventanas temporales alineadas (`lookback`: 90d macro, 252d técnico, 8 trimestres fundamentals). Targets multi-tarea: régimen discreto, probabilidad drawdown, retorno 20d.
- Versionado por `dataset_version` (`v1`, `v2`, …) y `index_version` para embeddings/noticias.

## Campos Clave por Entidad

### Macro Series (`silver/macro/*.parquet`)
- `date`
- `series_id`
- `value`
- `source` (FRED)
- `release_date`
- `frequency`
- `inflation_adjusted` (bool)

### Commodities (`silver/commodities/*.parquet`)
- `date`
- `symbol`
- `close`
- `open`, `high`, `low`, `volume`
- `currency`
- `exchange`

### FX Rates (`silver/fx/*.parquet`)
- `date`
- `pair`
- `close`
- `open`, `high`, `low`
- `volume` (cuando exista)
- `invert_rate` (bool que indica si se derivó del inverso)

### Equities & Technicals (`silver/equities/*.parquet`)
- `date`
- `ticker`
- `open`, `high`, `low`, `close`, `adj_close`, `volume`
- `sma_20/50/200`, `ema_20/50/200`
- `rsi_14`, `macd`, `macd_signal`, `atr_14`
- `drawdown`, `volatility_30d`

### Fundamentals (`silver/fundamentals/*.parquet`)
- `ticker`
- `fiscal_period`
- `revenue`, `revenue_growth`
- `ebit`, `ebit_margin`
- `net_income`, `net_margin`
- `roic`, `roiic`
- `free_cash_flow`, `fcf_yield`
- `capital_expenditures`
- `r_and_d`
- `total_debt`, `net_debt`
- `equity_beta`, `interest_coverage`

### Events & Regimes (`silver/events/*.parquet`)
- `event_id`
- `name`
- `type` (recession, depression, mania, crisis, policy)
- `start_date`, `end_date`
- `region`
- `severity` (ordinal)
- `description`
- `sources` (lista → JSON)

### Sentiment/News (`silver/news_sentiment/*.parquet`)
- `date`
- `ticker` / `macro_theme`
- `source`
- `headline`
- `sentiment_score`
- `relevance`
- `event_tags`

## Feature Engineering (Silver → Gold)

- **Macro**: rolling z-scores 90d, delta vs promedio histórico, flags de inversión de curva, gap vs inflación objetivo, shock en desempleo.
- **Commodities**: retornos 20/60/120d, spreads energía vs metales, ratio oro/petróleo, volatilidad (Parkinson).
- **FX**: índice de fortaleza USD, volatilidad (Garman-Klass) 30d, deviations vs PPP estimado.
- **Equities**: momentum multi-horizonte, factores de calidad (ROIC, ROIIC, márgenes), value (FCF yield, EV/EBITDA), riesgo (beta rolling, drawdown).
- **Eventos**: `regime_label` (0 normal, 1 recesión, 2 depresión, 3 crisis, 4 mania), `event_intensity`, proximidad a eventos.
- **Targets**:
  - `future_return_20d`
  - `future_drawdown_prob_20d`
  - `target_regime_next`

## Salidas Esperadas

- `data/gold/train.parquet`, `val.parquet`, `test.parquet` con columnas:
  - `date`, `ticker`
  - Prefijos de features: `macro_*`, `commodity_*`, `fx_*`, `tech_*`, `fund_*`, `event_*`, `sentiment_*`
  - Variables objetivo: `target_return_20d`, `target_drawdown_prob`, `target_regime`

## Próximos Pasos

1. Implementar ingestas Prefect por dominio siguiendo este esquema.
2. Normalizar y almacenar en `bronze/` con particiones por fuente/fecha.
3. Crear transformaciones en `src/caria/feature_store/transformations/` y actualizar `feature_store.yaml`.
4. Configurar validaciones (Great Expectations) para asegurar integridad y ausencia de lookahead.
5. Versionar datasets (`gold/v1`, `gold/v2`) y documentar cambios.

