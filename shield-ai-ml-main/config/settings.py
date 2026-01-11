from pydantic_settings import BaseSettings
from typing import Optional, List
from pydantic import validator
import os


class Settings(BaseSettings):
    """Application settings with environment variables"""
    
    # Application
    APP_NAME: str = "SHIELD AI Safety System"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "https://shield-safety.com"]
    
    # Database
    DATABASE_URL: str = "postgresql://shield:password@localhost:5432/shield_db"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AI/ML Models
    MODEL_CACHE_DIR: str = "storage/models"
    TRAINING_DATA_DIR: str = "data/processed"
    MODEL_VERSION: str = "v2.0"
    
    # External APIs
    MAPBOX_API_KEY: Optional[str] = None
    WEATHER_API_KEY: Optional[str] = None
    POLICE_API_URL: Optional[str] = None
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_PORT: int = 9090
    
    # Risk Thresholds
    LOW_RISK_THRESHOLD: float = 0.3
    MEDIUM_RISK_THRESHOLD: float = 0.6
    HIGH_RISK_THRESHOLD: float = 0.8
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        allowed = ["development", "testing", "production"]
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v
    
    @property
    def is_production(self):
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self):
        return self.ENVIRONMENT == "development"


settings = Settings()