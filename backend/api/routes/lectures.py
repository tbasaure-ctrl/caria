"""
API routes for recommended lectures/articles.
"""
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.services.lectures_service import get_lectures_service, LecturesService

router = APIRouter(prefix="/api/lectures", tags=["Lectures"])

class LectureRecommendation(BaseModel):
    title: str
    url: str
    source: str
    date: str

@router.get("/recommended", response_model=List[LectureRecommendation])
async def get_recommended_lectures(
    service: LecturesService = Depends(get_lectures_service)
) -> List[LectureRecommendation]:
    """
    Get daily recommended lectures/articles from external sources.
    """
    try:
        recommendations = service.get_daily_recommendations()
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
