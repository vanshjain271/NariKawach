from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from config.settings import settings

PUBLIC_PATHS = [
    "/", 
    "/health", 
    "/api/v1/health", 
    "/docs", 
    "/openapi.json"
]

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow public routes (no auth)
        if path in PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("x-api-key")
        if not api_key or api_key != settings.API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

        return await call_next(request)
