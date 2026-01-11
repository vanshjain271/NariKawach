from fastapi import APIRouter
from src.api.schemas import PredictRequest, PredictResponse
from src.risk_engine.risk_calculator import calculate_risk
from utils.logger import logger

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ML service running"}

@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    logger.info(f"Received prediction request for user {payload.user_id}")
    return calculate_risk(payload)
