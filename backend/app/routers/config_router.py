from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict
from app.services.feature_flags import get_feature_flags
router = APIRouter()
class FeaturesResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    appendicular_bones: bool
    teeth: bool
    trunk_cavities: bool
@router.get("/features", response_model=FeaturesResponse)
async def get_features() -> FeaturesResponse:
    flags = get_feature_flags()
    return FeaturesResponse(**flags.__dict__)
