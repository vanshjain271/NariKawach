from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import os

# Import internal modules
from ..decision_agent.rule_engine import RuleEngine
from ..decision_agent.intervention_logic import InterventionAgent
from ..decision_agent.priority_handler import PriorityHandler
from ..decision_agent.emergency_response import EmergencyResponseCoordinator
from ..risk_engine.risk_calculator import RiskCalculator
from ..risk_engine.ensemble_predictor import AdvancedRiskPredictor
from ..risk_engine.feature_engineering import SafetyFeatureEngineer
from ..anomaly_detection.route_anomaly import RouteAnomalyDetector
from ..anomaly_detection.stalking_detection import AdvancedStalkingDetector
from ..anomaly_detection.pattern_analyzer import PatternAnalyzer

from .middleware import SecurityMiddleware, RateLimitMiddleware
from .schemas import *
from ..utils.logger import setup_logger
from ..config.settings import settings


# Setup logger
logger = setup_logger(__name__)

# Security
security = HTTPBearer()

# Global instances
risk_calculator = None
risk_predictor = None
feature_engineer = None
route_detector = None
stalking_detector = None
pattern_analyzer = None
rule_engine = None
intervention_agent = None
priority_handler = None
emergency_coordinator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("Starting SHIELD AI Safety API...")
    
    # Initialize components
    await initialize_components()
    
    # Health check task
    asyncio.create_task(periodic_health_check())
    
    yield
    
    # Shutdown
    logger.info("Shutting down SHIELD AI Safety API...")
    await shutdown_components()


async def initialize_components():
    """Initialize all AI/ML components"""
    global risk_calculator, risk_predictor, feature_engineer
    global route_detector, stalking_detector, pattern_analyzer
    global rule_engine, intervention_agent, priority_handler, emergency_coordinator
    
    try:
        logger.info("Initializing AI/ML components...")
        
        # Risk Engine Components
        risk_calculator = RiskCalculator()
        risk_predictor = AdvancedRiskPredictor()
        feature_engineer = SafetyFeatureEngineer()
        
        # Anomaly Detection Components
        route_detector = RouteAnomalyDetector()
        stalking_detector = AdvancedStalkingDetector()
        pattern_analyzer = PatternAnalyzer()
        
        # Decision Agent Components
        rule_engine = RuleEngine()
        intervention_agent = InterventionAgent()
        priority_handler = PriorityHandler()
        emergency_coordinator = EmergencyResponseCoordinator()
        
        # Load models (if saved)
        await load_models()
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise


async def load_models():
    """Load pre-trained models"""
    try:
        models_dir = settings.MODEL_CACHE_DIR
        if os.path.exists(models_dir):
            # Load risk predictor models
            risk_predictor.load_models(models_dir)
            logger.info("Loaded pre-trained models")
    except Exception as e:
        logger.warning(f"Could not load models: {e}")


async def shutdown_components():
    """Shutdown and cleanup components"""
    try:
        # Save models
        models_dir = settings.MODEL_CACHE_DIR
        risk_predictor.save_models(models_dir)
        logger.info("Saved models before shutdown")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


async def periodic_health_check():
    """Periodic health check task"""
    while True:
        try:
            # Check component health
            health_status = await check_component_health()
            
            if not health_status.get('overall_healthy', False):
                logger.warning(f"Health check failed: {health_status}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            await asyncio.sleep(60)


async def check_component_health() -> Dict:
    """Check health of all components"""
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'components': {},
        'overall_healthy': True
    }
    
    components = [
        ('risk_calculator', risk_calculator),
        ('risk_predictor', risk_predictor),
        ('feature_engineer', feature_engineer),
        ('route_detector', route_detector),
        ('stalking_detector', stalking_detector),
        ('pattern_analyzer', pattern_analyzer),
        ('rule_engine', rule_engine),
        ('intervention_agent', intervention_agent),
        ('priority_handler', priority_handler),
        ('emergency_coordinator', emergency_coordinator)
    ]
    
    for name, component in components:
        try:
            if component is None:
                health_status['components'][name] = {
                    'status': 'not_initialized',
                    'healthy': False
                }
                health_status['overall_healthy'] = False
            else:
                # Basic health check
                health_status['components'][name] = {
                    'status': 'initialized',
                    'healthy': True
                }
        except Exception as e:
            health_status['components'][name] = {
                'status': 'error',
                'error': str(e),
                'healthy': False
            }
            health_status['overall_healthy'] = False
    
    return health_status


# Create FastAPI app
app = FastAPI(
    title="SHIELD AI Safety API",
    description="Advanced AI/ML backend for women's safety system",
    version="2.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(SecurityMiddleware)
app.add_middleware(RateLimitMiddleware)


# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    
    # In production, validate against database or auth service
    if not token.startswith("shield_"):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token


# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "path": request.url.path,
            "timestamp": datetime.now().isoformat(),
            "request_id": request.state.request_id if hasattr(request.state, 'request_id') else None
        }
    )


# Import and include routers
from .endpoints import router as api_router
app.include_router(api_router, prefix=settings.API_PREFIX)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "SHIELD AI Safety API",
        "version": "2.0.0",
        "status": "operational",
        "environment": settings.ENVIRONMENT,
        "documentation": "/docs" if settings.DEBUG else None,
        "endpoints": [
            f"{settings.API_PREFIX}/health",
            f"{settings.API_PREFIX}/risk/assess",
            f"{settings.API_PREFIX}/anomalies/detect",
            f"{settings.API_PREFIX}/stalking/check",
            f"{settings.API_PREFIX}/intervention/decide",
            f"{settings.API_PREFIX}/emergency/activate"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = await check_component_health()
        
        return {
            "status": "healthy" if health_status['overall_healthy'] else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": health_status['components'],
            "environment": settings.ENVIRONMENT,
            "version": "2.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


# Development endpoints (only in debug mode)
if settings.DEBUG:
    @app.get("/debug/components")
    async def debug_components(token: str = Depends(verify_token)):
        """Debug endpoint to check component status"""
        return {
            "components": {
                "risk_calculator": risk_calculator is not None,
                "risk_predictor": risk_predictor is not None,
                "feature_engineer": feature_engineer is not None,
                "route_detector": route_detector is not None,
                "stalking_detector": stalking_detector is not None,
                "pattern_analyzer": pattern_analyzer is not None,
                "rule_engine": rule_engine is not None,
                "intervention_agent": intervention_agent is not None,
                "priority_handler": priority_handler is not None,
                "emergency_coordinator": emergency_coordinator is not None
            },
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/debug/test-emergency")
    async def test_emergency_protocol(
        test_request: Dict,
        token: str = Depends(verify_token)
    ):
        """Test emergency protocol"""
        try:
            user_id = test_request.get('user_id', 'test_user')
            protocol_type = test_request.get('protocol_type', 'stalking_emergency')
            
            result = emergency_coordinator.test_emergency_protocol(user_id, protocol_type)
            
            return {
                "test_result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )