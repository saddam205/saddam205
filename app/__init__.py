"""
AI Trading Bot Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="AI Trading Bot",
    description="Advanced AI-powered cryptocurrency trading bot",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
try:
    from app.api.routes import router as api_router
    app.include_router(api_router, prefix="/api/v1", tags=["Trading API"])
    print("✅ API Routes loaded successfully")
except Exception as e:
    print(f"❌ Error loading API routes: {e}")

try:
    from app.api.websocket import router as ws_router
    app.include_router(ws_router, prefix="/ws", tags=["WebSocket"])
    print("✅ WebSocket routes loaded successfully")
except Exception as e:
    print(f"❌ Error loading WebSocket routes: {e}")

@app.get("/")
async def root():
    return {
        "name": "AI Trading Bot",
        "version": "1.0.0",
        "status": "running",
        "mode": "VIRTUAL",
        "endpoints": {
            "api": "/api/v1",
            "docs": "/docs",
            "health": "/health",
            "websocket": "ws://localhost:8000/ws"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "mode": "VIRTUAL", "timestamp": "2024-01-01T00:00:00Z"}

# Also add root level endpoints for testing
@app.get("/api/v1/balance")
async def balance_root():
    """Test endpoint for balance"""
    return {"balance": 10000.0, "total_pnl": 0.0, "positions": 0}
