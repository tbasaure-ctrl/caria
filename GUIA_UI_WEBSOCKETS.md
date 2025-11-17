# Guía: Conectar UI con la API - Precios en Tiempo Real

## Opciones para Precios en Tiempo Real

### Opción 1: Polling Simple (Más Fácil - Recomendado para empezar)

**Ventajas:**
- ✅ Fácil de implementar
- ✅ No requiere configuración adicional
- ✅ Funciona con HTTP estándar

**Desventajas:**
- ⚠️ Menos eficiente (hace requests periódicos)
- ⚠️ Puede tener delay de actualización

**Implementación:**

```typescript
// En tu componente React (MarketIndices.tsx o Portfolio.tsx)
import { useEffect, useState } from 'react';
import { fetchWithAuth } from '../services/apiService';

const POLLING_INTERVAL = 30000; // 30 segundos

function useRealtimePrices(tickers: string[]) {
  const [prices, setPrices] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPrices = async () => {
      try {
        const response = await fetchWithAuth('/api/prices/realtime', {
          method: 'POST',
          body: JSON.stringify({ tickers }),
        });
        const data = await response.json();
        setPrices(data.prices);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching prices:', error);
      }
    };

    // Fetch inmediatamente
    fetchPrices();

    // Luego cada 30 segundos
    const interval = setInterval(fetchPrices, POLLING_INTERVAL);

    return () => clearInterval(interval);
  }, [tickers]);

  return { prices, loading };
}
```

### Opción 2: WebSockets (Más Eficiente - Para producción)

**Ventajas:**
- ✅ Actualizaciones en tiempo real instantáneas
- ✅ Más eficiente (una conexión persistente)
- ✅ Mejor experiencia de usuario

**Desventajas:**
- ⚠️ Requiere configuración adicional
- ⚠️ Más complejo de implementar
- ⚠️ Necesita manejar reconexiones

## Implementación: Polling Simple (Recomendado)

### Paso 1: Actualizar apiService.ts

```typescript
// services/apiService.ts

export const fetchPrices = async (tickers: string[]): Promise<Record<string, any>> => {
  const response = await fetchWithAuth(`${API_URL}/api/prices/realtime`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tickers }),
  });
  
  if (!response.ok) {
    throw new Error('Error fetching prices');
  }
  
  const data = await response.json();
  return data.prices;
};

export const fetchHoldingsWithPrices = async (): Promise<any> => {
  const response = await fetchWithAuth(`${API_URL}/api/holdings/with-prices`);
  
  if (!response.ok) {
    throw new Error('Error fetching holdings');
  }
  
  return response.json();
};
```

### Paso 2: Actualizar MarketIndices.tsx

```typescript
import React, { useEffect, useState } from 'react';
import { fetchPrices } from '../../services/apiService';
import { ArrowUpIcon, ArrowDownIcon } from '../Icons';
import { WidgetCard } from './WidgetCard';

const MARKET_INDICES = ['SPY', 'QQQ', 'DIA', 'IWM']; // ETFs principales

export const MarketIndices: React.FC = () => {
  const [prices, setPrices] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const updatePrices = async () => {
      try {
        const data = await fetchPrices(MARKET_INDICES);
        setPrices(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching market indices:', error);
        setLoading(false);
      }
    };

    // Actualizar inmediatamente
    updatePrices();

    // Actualizar cada 30 segundos
    const interval = setInterval(updatePrices, 30000);

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <WidgetCard title="MARKET OVERVIEW">
        <div className="text-slate-400 text-sm">Cargando...</div>
      </WidgetCard>
    );
  }

  return (
    <WidgetCard title="MARKET OVERVIEW">
      <div className="space-y-2">
        {MARKET_INDICES.map(ticker => {
          const priceData = prices[ticker];
          if (!priceData) return null;

          const value = priceData.price || priceData.previousClose || 0;
          const change = priceData.change || 0;
          const changePercent = priceData.changesPercentage || 0;
          const isPositive = change >= 0;

          return (
            <div key={ticker} className="flex justify-between items-center text-sm">
              <span className="text-slate-200">{ticker}</span>
              <div className="text-right">
                <span className="font-mono">${value.toFixed(2)}</span>
                <span className={`ml-2 font-semibold flex items-center justify-end ${isPositive ? 'text-blue-400' : 'text-red-400'}`}>
                  {isPositive ? <ArrowUpIcon className="w-3 h-3 mr-1"/> : <ArrowDownIcon className="w-3 h-3 mr-1"/>}
                  {changePercent.toFixed(2)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </WidgetCard>
  );
};
```

### Paso 3: Actualizar Portfolio.tsx

```typescript
import React, { useState, useEffect } from 'react';
import { fetchHoldingsWithPrices } from '../../services/apiService';
import { WidgetCard } from './WidgetCard';

export const Portfolio: React.FC<{ id?: string }> = ({ id }) => {
  const [portfolioData, setPortfolioData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const updatePortfolio = async () => {
      try {
        const data = await fetchHoldingsWithPrices();
        setPortfolioData(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching portfolio:', error);
        setLoading(false);
      }
    };

    updatePortfolio();
    const interval = setInterval(updatePortfolio, 30000); // 30 segundos

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <WidgetCard id={id} title="PORTFOLIO SNAPSHOT">
        <div className="text-slate-400 text-sm">Cargando...</div>
      </WidgetCard>
    );
  }

  if (!portfolioData || portfolioData.holdings.length === 0) {
    return (
      <WidgetCard id={id} title="PORTFOLIO SNAPSHOT">
        <div className="text-slate-400 text-sm">No hay holdings. Agrega posiciones para ver tu portfolio.</div>
      </WidgetCard>
    );
  }

  return (
    <WidgetCard id={id} title="PORTFOLIO SNAPSHOT">
      <div className="space-y-4">
        <div>
          <h4 className="text-xs text-slate-400 mb-2">Resumen</h4>
          <div className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-slate-300">Valor Total:</span>
              <span className="font-mono text-slate-100">${portfolioData.total_value.toFixed(2)}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-300">Costo Total:</span>
              <span className="font-mono text-slate-300">${portfolioData.total_cost.toFixed(2)}</span>
            </div>
            <div className={`flex justify-between text-sm ${portfolioData.total_gain_loss >= 0 ? 'text-blue-400' : 'text-red-400'}`}>
              <span>Ganancia/Pérdida:</span>
              <span className="font-mono">
                ${portfolioData.total_gain_loss.toFixed(2)} ({portfolioData.total_gain_loss_pct.toFixed(2)}%)
              </span>
            </div>
          </div>
        </div>
        
        <div>
          <h4 className="text-xs text-slate-400 mb-2">Posiciones</h4>
          <div className="space-y-2">
            {portfolioData.holdings.map((holding: any) => (
              <div key={holding.ticker} className="flex justify-between items-center text-sm border-b border-slate-800 pb-2">
                <div>
                  <span className="text-slate-200 font-semibold">{holding.ticker}</span>
                  <span className="text-slate-400 ml-2">{holding.quantity} acciones</span>
                </div>
                <div className="text-right">
                  <div className="font-mono text-slate-100">${holding.current_price.toFixed(2)}</div>
                  <div className={`text-xs ${holding.gain_loss >= 0 ? 'text-blue-400' : 'text-red-400'}`}>
                    {holding.gain_loss >= 0 ? '+' : ''}{holding.gain_loss_pct.toFixed(2)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </WidgetCard>
  );
};
```

## Implementación: WebSockets (Opcional - Avanzado)

Si quieres implementar WebSockets más adelante, necesitarías:

1. **Backend**: Agregar soporte WebSocket a FastAPI
2. **Frontend**: Usar `useWebSocket` hook o librería como `socket.io-client`
3. **Manejo de reconexión**: Implementar lógica para reconectar si se pierde la conexión

**Ejemplo básico con FastAPI WebSockets:**

```python
# En app.py
from fastapi import WebSocket

@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Enviar precios cada X segundos
            prices = get_latest_prices()
            await websocket.send_json(prices)
            await asyncio.sleep(5)  # Actualizar cada 5 segundos
    except WebSocketDisconnect:
        pass
```

## Recomendación

**Para empezar:** Usa **Polling Simple** (Opción 1)
- Es más fácil de implementar
- Funciona inmediatamente
- Puedes migrar a WebSockets después si lo necesitas

**Para producción a gran escala:** Considera **WebSockets**
- Si tienes muchos usuarios simultáneos
- Si necesitas actualizaciones instantáneas
- Si quieres reducir carga en el servidor

## Próximos Pasos

1. ✅ Implementar polling simple en MarketIndices
2. ✅ Implementar polling simple en Portfolio
3. ✅ Agregar manejo de errores y estados de carga
4. ⏭️ (Opcional) Migrar a WebSockets si es necesario

