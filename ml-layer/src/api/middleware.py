from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from config.settings import settings

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("x-api-key")
        if api_key != settings.API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return await call_next(request)
