from fastapi import FastAPI
from src.api.endpoints import router
from src.api.middleware import AuthMiddleware

app = FastAPI(title="Nari Kawach ML Service")

# ✅ Public health route (NO auth)
@app.get("/health")
def health():
    return {"status": "ok"}

# Middleware
app.add_middleware(AuthMiddleware)

# API routes
from config.settings import settings

app.include_router(router, prefix=settings.API_PREFIX)
