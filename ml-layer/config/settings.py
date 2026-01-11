from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "Nari Kawach ML Service"
    API_KEY: str = "nari_kawach_secret"
    RATE_LIMIT: int = 100

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
