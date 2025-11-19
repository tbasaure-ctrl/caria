"""
Fear and Greed Index endpoint.
Fetches CNN Fear and Greed Index data from Alternative.me API.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("caria.api.fear_greed")

router = APIRouter(prefix="/api/market", tags=["Market Data"])


class FearGreedResponse(BaseModel):
    """Response model for Fear and Greed Index."""
    value: int = Field(..., ge=0, le=100, description="Current Fear and Greed Index value (0-100)")
    classification: str = Field(..., description="Classification: Extreme Fear, Fear, Neutral, Greed, Extreme Greed")
    timestamp: str = Field(..., description="Timestamp of the data")
    previous_close: Optional[int] = Field(None, ge=0, le=100, description="Previous day's value")
    change: Optional[int] = Field(None, description="Change from previous day")


@router.get("/fear-greed", response_model=FearGreedResponse)
async def get_fear_greed_index() -> FearGreedResponse:
    """
    Get CNN Fear and Greed Index in real-time.
    
    Uses Alternative.me API which provides free access to Fear and Greed Index data.
    The index ranges from 0-100:
    - 0-25: Extreme Fear
    - 25-45: Fear
    - 45-55: Neutral
    - 55-75: Greed
    - 75-100: Extreme Greed
    """
    try:
        # Alternative.me API endpoint for Fear and Greed Index
        api_url = "https://api.alternative.me/fng/"
        
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("data") or len(data["data"]) == 0:
            raise HTTPException(status_code=503, detail="Fear and Greed Index data not available")
        
        # Get the most recent value (first in the array)
        current = data["data"][0]
        value = int(current["value"])
        
        # Get previous value if available
        previous_value = None
        if len(data["data"]) > 1:
            previous_value = int(data["data"][1]["value"])
        
        # Classify the value
        if value <= 25:
            classification = "Extreme Fear"
        elif value <= 45:
            classification = "Fear"
        elif value <= 55:
            classification = "Neutral"
        elif value <= 75:
            classification = "Greed"
        else:
            classification = "Extreme Greed"
        
        change = None
        if previous_value is not None:
            change = value - previous_value
        
        return FearGreedResponse(
            value=value,
            classification=classification,
            timestamp=current.get("timestamp", ""),
            previous_close=previous_value,
            change=change,
        )
        
    except requests.RequestException as e:
        LOGGER.exception(f"Error fetching Fear and Greed Index: {e}")
        raise HTTPException(status_code=503, detail="Failed to fetch Fear and Greed Index data") from e
    except (ValueError, KeyError, IndexError) as e:
        LOGGER.exception(f"Error parsing Fear and Greed Index data: {e}")
        raise HTTPException(status_code=500, detail="Invalid Fear and Greed Index data format") from e

