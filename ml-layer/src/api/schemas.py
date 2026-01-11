from pydantic import BaseModel
from typing import List

class Location(BaseModel):
    lat: float
    lng: float

class TimeContext(BaseModel):
    hour: int
    day: str

class PredictRequest(BaseModel):
    user_id: str
    current_location: Location
    route: List[Location]
    time: TimeContext
    crime_density: float
    crowd_density: float
    lighting_score: float
    speed: float
    route_deviation_score: float

class PredictResponse(BaseModel):
    risk_score: float
    risk_level: str
    confidence: float
    reason: str
