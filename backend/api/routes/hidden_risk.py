from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_current_user, get_db_connection
from api.services.hidden_risk_service import hidden_risk_service
from caria.models.auth import UserInDB

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

@router.post("/hidden-risk")
async def get_hidden_risk_report(
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db_connection)
):
    """
    Generates a hidden risk report for the user's portfolio.
    Combines Hydraulic Stack, Caria Cortex, and News Sentiment.
    """
    try:
        report = await hidden_risk_service.generate_report(current_user.id, db)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


