"""Rutas para holdings (posiciones) de usuarios."""

from __future__ import annotations

import sys
from pathlib import Path

# Configurar paths para encontrar caria
CURRENT_FILE = Path(__file__).resolve()
CARIA_DATA_SRC = CURRENT_FILE.parent.parent.parent.parent / "caria_data" / "src"
if CARIA_DATA_SRC.exists() and str(CARIA_DATA_SRC) not in sys.path:
    sys.path.insert(0, str(CARIA_DATA_SRC))

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_current_user, get_db_connection
from caria.models.auth import UserInDB
from api.services.alpha_vantage_client import alpha_vantage_client

router = APIRouter(prefix="/api/holdings", tags=["holdings"])


class HoldingCreate(BaseModel):
    """Request para crear/actualizar holding."""
    ticker: str = Field(..., min_length=1, max_length=10, description="Símbolo del ticker (use 'CASH' for cash position)")
    quantity: float = Field(..., ge=0, description="Cantidad de acciones (or cash amount)")
    average_cost: float = Field(..., ge=0, description="Costo promedio por acción (or 1.0 for cash)")
    notes: str | None = Field(None, description="Notas adicionales")


class HoldingResponse(BaseModel):
    """Respuesta con datos de holding."""
    id: UUID
    ticker: str
    quantity: float
    average_cost: float
    notes: str | None
    created_at: str
    updated_at: str


class HoldingsWithPricesResponse(BaseModel):
    """Respuesta con holdings y precios en tiempo real."""
    holdings: list[dict[str, Any]]
    total_value: float
    total_cost: float
    total_gain_loss: float
    total_gain_loss_pct: float




@router.get("", response_model=list[HoldingResponse])
def get_holdings(
    current_user: UserInDB = Depends(get_current_user),
    conn = Depends(get_db_connection),
) -> list[HoldingResponse]:
    """Obtiene todos los holdings del usuario actual."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, ticker, quantity, average_cost, notes, 
                       created_at, updated_at
                FROM holdings
                WHERE user_id = %s
                ORDER BY ticker
                """,
                (str(current_user.id),),
            )
            rows = cur.fetchall()
            
            holdings = []
            for row in rows:
                holdings.append(HoldingResponse(
                    id=row[0],
                    ticker=row[1],
                    quantity=float(row[2]),
                    average_cost=float(row[3]),
                    notes=row[4],
                    created_at=row[5].isoformat() if row[5] else "",
                    updated_at=row[6].isoformat() if row[6] else "",
                ))
            
            return holdings
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo holdings: {str(exc)}",
        ) from exc


@router.post("", response_model=HoldingResponse, status_code=status.HTTP_201_CREATED)
def create_holding(
    holding: HoldingCreate,
    current_user: UserInDB = Depends(get_current_user),
    conn = Depends(get_db_connection),
) -> HoldingResponse:
    """Crea o actualiza un holding del usuario."""
    try:
        with conn.cursor() as cur:
            # Intentar insertar o actualizar (UPSERT)
            cur.execute(
                """
                INSERT INTO holdings (user_id, ticker, quantity, average_cost, notes)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id, ticker) 
                DO UPDATE SET 
                    quantity = EXCLUDED.quantity,
                    average_cost = EXCLUDED.average_cost,
                    notes = EXCLUDED.notes,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id, ticker, quantity, average_cost, notes, created_at, updated_at
                """,
                (
                    str(current_user.id),
                    holding.ticker.upper(),
                    holding.quantity,
                    holding.average_cost,
                    holding.notes,
                ),
            )
            row = cur.fetchone()
            conn.commit()
            
            return HoldingResponse(
                id=row[0],
                ticker=row[1],
                quantity=float(row[2]),
                average_cost=float(row[3]),
                notes=row[4],
                created_at=row[5].isoformat() if row[5] else "",
                updated_at=row[6].isoformat() if row[6] else "",
            )
    except Exception as exc:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creando holding: {str(exc)}",
        ) from exc


class HoldingUpdate(BaseModel):
    """Request para actualizar holding parcialmente."""
    quantity: float | None = Field(None, ge=0, description="Cantidad de acciones")
    average_cost: float | None = Field(None, ge=0, description="Costo promedio por acción")
    notes: str | None = Field(None, description="Notas adicionales")


@router.patch("/{holding_id}", response_model=HoldingResponse)
def update_holding(
    holding_id: UUID,
    updates: HoldingUpdate,
    current_user: UserInDB = Depends(get_current_user),
    conn = Depends(get_db_connection),
) -> HoldingResponse:
    """Actualiza parcialmente un holding del usuario."""
    try:
        # Build dynamic update query based on provided fields
        update_fields = []
        values = []
        
        if updates.quantity is not None:
            update_fields.append("quantity = %s")
            values.append(updates.quantity)
        if updates.average_cost is not None:
            update_fields.append("average_cost = %s")
            values.append(updates.average_cost)
        if updates.notes is not None:
            update_fields.append("notes = %s")
            values.append(updates.notes)
        
        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update provided",
            )
        
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        values.extend([str(holding_id), str(current_user.id)])
        
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE holdings
                SET {', '.join(update_fields)}
                WHERE id = %s AND user_id = %s
                RETURNING id, ticker, quantity, average_cost, notes, created_at, updated_at
                """,
                tuple(values),
            )
            row = cur.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Holding no encontrado",
                )
            
            conn.commit()
            
            return HoldingResponse(
                id=row[0],
                ticker=row[1],
                quantity=float(row[2]),
                average_cost=float(row[3]),
                notes=row[4],
                created_at=row[5].isoformat() if row[5] else "",
                updated_at=row[6].isoformat() if row[6] else "",
            )
    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error actualizando holding: {str(exc)}",
        ) from exc


@router.delete("/{holding_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_holding(
    holding_id: UUID,
    current_user: UserInDB = Depends(get_current_user),
    conn = Depends(get_db_connection),
) -> None:
    """Elimina un holding del usuario."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM holdings
                WHERE id = %s AND user_id = %s
                """,
                (str(holding_id), str(current_user.id)),
            )
            if cur.rowcount == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Holding no encontrado",
                )
            conn.commit()
    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error eliminando holding: {str(exc)}",
        ) from exc


@router.get("/with-prices", response_model=HoldingsWithPricesResponse)
def get_holdings_with_prices(
    currency: str = "USD",
    current_user: UserInDB = Depends(get_current_user),
    conn = Depends(get_db_connection),
) -> HoldingsWithPricesResponse:
    """Get user holdings with real-time prices.
    
    Calculates total value, gain/loss and percentages.
    Falls back to average cost if real-time price unavailable.
    Uses Alpha Vantage for Crypto (tickers like BTC, ETH, BTCUSD).
    """
    import logging
    logger = logging.getLogger("caria.api.holdings")
    
    try:
        # Get holdings from database
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, ticker, quantity, average_cost, notes
                FROM holdings
                WHERE user_id = %s
                ORDER BY ticker
                """,
                (str(current_user.id),),
            )
            rows = cur.fetchall()
        
        if not rows:
            return HoldingsWithPricesResponse(
                holdings=[],
                total_value=0.0,
                total_cost=0.0,
                total_gain_loss=0.0,
                total_gain_loss_pct=0.0,
            )
        
        # Separate tickers
        crypto_tickers = []
        stock_tickers = []
        
        # Heuristic: If ticker is 3-4 chars and in common list OR contains 'USD', treat as crypto
        # Or if user explicitly marked it (we don't have that flag yet)
        # Simple list for now + USD suffix check
        known_crypto = {'BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'DOGE', 'DOT', 'LINK'}
        
        for row in rows:
            ticker = row[1].upper()
            if ticker in known_crypto or ticker.endswith('USD') or ticker.endswith('-USD'):
                crypto_tickers.append(ticker)
            else:
                stock_tickers.append(ticker)
        
        logger.info(f"Fetching prices for Stocks: {stock_tickers}, Crypto: {crypto_tickers}")
        
        prices: dict = {}
        
        # 1. Fetch Stocks from FMP
        if stock_tickers:
            try:
                from caria.ingestion.clients.fmp_client import FMPClient
                fmp_client = FMPClient()
                stock_prices = fmp_client.get_realtime_prices_batch(stock_tickers)
                prices.update(stock_prices)
            except Exception as fmp_exc:
                logger.warning(f"FMP price fetch failed: {fmp_exc}")

        # 2. Fetch Crypto from Alpha Vantage
        for ticker in crypto_tickers:
            try:
                # Normalize ticker for AV (needs 'BTC' for 'BTCUSD' usually, or explicit pair)
                # If it's just 'BTC', assume 'USD' market
                symbol = ticker.replace('USD', '').replace('-', '')
                data = alpha_vantage_client.get_crypto_price(symbol=symbol, market='USD')
                if data:
                    prices[ticker] = {
                        "price": data["price"],
                        "change": 0, # AV realtime rate doesn't give 24h change directly in this endpoint easily without daily series
                        "changesPercentage": 0
                    }
            except Exception as av_exc:
                logger.warning(f"AV crypto fetch failed for {ticker}: {av_exc}")
        
        # Get FX rate for currency conversion
        fx_rate = 1.0
        if currency.upper() == "CLP":
            try:
                from caria.ingestion.clients.fmp_client import FMPClient
                fmp_client = FMPClient()
                fx_data = fmp_client.get_realtime_prices_batch(["USDCLP=X"])
                if fx_data and "USDCLP=X" in fx_data:
                    fx_rate = float(fx_data["USDCLP=X"].get("price", 1.0))
                else:
                    # Fallback to approximate rate
                    fx_rate = 950.0
                    logger.warning("Could not fetch USD/CLP rate, using fallback")
            except Exception as fx_exc:
                logger.warning(f"FX rate fetch failed: {fx_exc}, using fallback")
                fx_rate = 950.0
        
        # Calculate values
        holdings_data = []
        total_value = 0.0
        total_cost = 0.0
        
        for row in rows:
            holding_id, ticker, quantity, avg_cost, notes = row[0], row[1], float(row[2]), float(row[3]), row[4]
            
            # Handle cash position
            if ticker.upper() == "CASH":
                current_price = 1.0  # Cash is always 1:1
                # Cash is stored in USD (quantity is the USD amount)
                # avg_cost should always be 1.0 for cash
                cash_amount_usd = quantity
                cash_cost_basis_usd = cash_amount_usd * avg_cost  # Should be quantity * 1.0 = quantity
                
                # Convert to target currency
                if currency.upper() == "CLP":
                    current_value = cash_amount_usd * fx_rate
                    cost_basis = cash_cost_basis_usd * fx_rate
                else:
                    current_value = cash_amount_usd
                    cost_basis = cash_cost_basis_usd
                
                total_value += current_value
                total_cost += cost_basis
                gain_loss = 0.0  # Cash doesn't have gain/loss
                gain_loss_pct = 0.0
                price_change = 0.0
                price_change_pct = 0.0
                price_source = "cash"
            else:
                # Calculate cost basis in USD first
                cost_basis_usd = quantity * avg_cost
                
                price_data = prices.get(ticker, {})
                
                # Try multiple fields for current price with fallback to average cost
                current_price_usd = (
                    price_data.get("price") or 
                    price_data.get("previousClose") or
                    price_data.get("close") or
                    avg_cost  # Fallback to average cost if no price available
                )
                
                # Ensure we have a valid number
                try:
                    current_price_usd = float(current_price_usd) if current_price_usd else avg_cost
                except (ValueError, TypeError):
                    current_price_usd = avg_cost
                
                # Convert to target currency
                current_price = current_price_usd * fx_rate
                current_value = quantity * current_price
                cost_basis = cost_basis_usd * fx_rate
                
                # Calculate gain/loss in target currency
                gain_loss = current_value - cost_basis
                gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0.0
                price_change = (price_data.get("change", 0.0) or 0.0) * fx_rate
                price_change_pct = price_data.get("changesPercentage", 0.0) or 0.0
                price_source = "live" if price_data.get("price") else "fallback"
                
                # Add to totals AFTER conversion
                total_value += current_value
                total_cost += cost_basis
            
            holdings_data.append({
                "id": str(holding_id),
                "ticker": ticker,
                "quantity": quantity,
                "average_cost": avg_cost,
                "notes": notes,
                "current_price": current_price,
                "cost_basis": cost_basis,
                "current_value": current_value,
                "gain_loss": gain_loss,
                "gain_loss_pct": gain_loss_pct,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "price_source": price_source,
            })
        
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0.0
        
        return HoldingsWithPricesResponse(
            holdings=holdings_data,
            total_value=total_value,
            total_cost=total_cost,
            total_gain_loss=total_gain_loss,
            total_gain_loss_pct=total_gain_loss_pct,
        )
    except Exception as exc:
        import traceback
        logger.error(f"Error fetching holdings with prices: {exc}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching holdings with prices: {str(exc)}",
        ) from exc
