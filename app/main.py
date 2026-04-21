"""
main.py
FastAPI application entry point for AI Trading Bot.
"""

import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import config
from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.api.middleware import setup_middleware
from app.core.trading_engine import trading_engine
from app.utils.logger import setup_logger


# -------------------------------
# 🔧 Setup Logging
# -------------------------------
setup_logger()
logger = logging.getLogger(__name__)


# -------------------------------
# 📁 Base Directory (robust paths)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend", "public")


# -------------------------------
# 🔄 Lifespan Manager
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown handler"""

    # 🚀 Startup
    logger.info("=" * 60)
    logger.info("🚀 Starting AI Trading Bot")
    logger.info("=" * 60)
    logger.info(f"Version: {app.version}")
    logger.info(f"Trading Mode: {config.TRADING_MODE}")
    logger.info(f"Supported Symbols: {len(config.SUPPORTED_SYMBOLS)}")

    # Start trading engine
    if config.TRADING_MODE.upper() != "MANUAL":
        try:
            await trading_engine.start()
            logger.info("✅ Trading engine started")
        except Exception as e:
            logger.error(f"❌ Failed to start trading engine: {e}")

    # Config warnings
    try:
        warnings = config.validate()
        for warning in warnings:
            logger.warning(f"⚠️ {warning}")
    except Exception as e:
        logger.error(f"Config validation failed: {e}")

    yield

    # 🛑 Shutdown
    logger.info("=" * 60)
    logger.info("🛑 Shutting down AI Trading Bot")
    logger.info("=" * 60)

    try:
        await trading_engine.stop()
        logger.info("✅ Trading engine stopped")
    except Exception as e:
        logger.error(f"❌ Error stopping trading engine: {e}")


# -------------------------------
# 🚀 Create FastAPI App
# -------------------------------
app = FastAPI(
    title="AI Trading Bot API",
    description="Advanced AI-powered cryptocurrency trading bot",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# -------------------------------
# 📦 Static Files (safe path)
# -------------------------------
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
else:
    logger.warning(f"Frontend directory not found: {FRONTEND_DIR}")


# -------------------------------
# 🌐 CORS (dev + prod safe)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# ⚙️ Custom Middleware
# -------------------------------
setup_middleware(app)


# -------------------------------
# 🔌 Routers
# -------------------------------
app.include_router(api_router, prefix="/api/v1", tags=["Trading"])
app.include_router(ws_router, prefix="/ws", tags=["WebSocket"])


# -------------------------------
# 🏠 Root Endpoint
# -------------------------------
@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "AI Trading Bot",
        "version": app.version,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# -------------------------------
# ❤️ Health Check
# -------------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "version": app.version,
        "trading_mode": config.TRADING_MODE,
        "timestamp": datetime.now().isoformat()
    }


# -------------------------------
# 📊 Metrics Endpoint
# -------------------------------
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    import psutil

    disk_path = os.getcwd()

    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage_percent": psutil.disk_usage(disk_path).percent,
        "trading_mode": config.TRADING_MODE
    }


# -------------------------------
# 🖥️ Dashboard
# -------------------------------
@app.get("/dashboard", tags=["Frontend"])
async def dashboard():
    dashboard_path = os.path.join(FRONTEND_DIR, "dashboard.html")

    if not os.path.exists(dashboard_path):
        return JSONResponse(
            status_code=404,
            content={"error": "Dashboard not found"}
        )

    return FileResponse(dashboard_path)


# -------------------------------
# ❗ Exception Handlers
# -------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if config.LOG_LEVEL == "DEBUG" else None
        }
    )