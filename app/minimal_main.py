"""
Minimal FastAPI application for AI Trading Bot
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Trading Bot API",
    description="Advanced AI-powered cryptocurrency trading bot",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://192.168.3.157:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class TradeRequest(BaseModel):
    symbol: str
    quantity: float
    side: str  # buy or sell

class TradeResponse(BaseModel):
    status: str
    trade_id: str
    symbol: str
    quantity: float
    side: str
    price: float
    timestamp: str
    message: str

# Endpoints
@app.get("/")
async def root():
    return {
        "status": "online",
        "name": "AI Trading Bot",
        "version": "2.0.0",
        "documentation": "/docs",
        "health": "/health",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "trading_bot",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/status")
async def get_status():
    return {
        "status": "running",
        "mode": "paper_trading",
        "active_strategies": ["momentum", "mean_reversion"],
        "portfolio_value": 100000.0,
        "open_positions": 0,
        "daily_pnl": 1250.0
    }

@app.post("/api/v1/trade", response_model=TradeResponse)
async def execute_trade(trade: TradeRequest):
    logger.info(f"Trade request: {trade.dict()}")
    
    # Simulate price
    import random
    simulated_price = 50000.0 + random.uniform(-1000, 1000)
    
    return TradeResponse(
        status="success",
        trade_id=f"trade_{int(datetime.utcnow().timestamp())}",
        symbol=trade.symbol,
        quantity=trade.quantity,
        side=trade.side,
        price=simulated_price,
        timestamp=datetime.utcnow().isoformat(),
        message=f"{trade.side.upper()} order executed successfully"
    )

@app.get("/api/v1/portfolio")
async def get_portfolio():
    return {
        "total_value": 100000.0,
        "cash": 75000.0,
        "positions": [
            {
                "symbol": "BTC/USDT",
                "quantity": 0.5,
                "avg_price": 48000.0,
                "current_price": 50000.0,
                "value": 25000.0,
                "pnl": 1000.0,
                "pnl_percentage": 4.17
            },
            {
                "symbol": "ETH/USDT",
                "quantity": 5.0,
                "avg_price": 3000.0,
                "current_price": 3100.0,
                "value": 15500.0,
                "pnl": 500.0,
                "pnl_percentage": 3.33
            }
        ],
        "daily_pnl": 1250.0,
        "total_pnl": 15000.0,
        "roi_percentage": 17.65
    }

@app.get("/api/v1/strategies")
async def get_strategies():
    return {
        "strategies": [
            {
                "id": 1,
                "name": "momentum",
                "display_name": "Momentum Strategy",
                "enabled": True,
                "risk_level": "medium",
                "parameters": {"period": 14, "threshold": 2.0}
            },
            {
                "id": 2,
                "name": "mean_reversion",
                "display_name": "Mean Reversion",
                "enabled": True,
                "risk_level": "low",
                "parameters": {"lookback": 20, "std_dev": 2.0}
            },
            {
                "id": 3,
                "name": "ml_predictor",
                "display_name": "ML Price Predictor",
                "enabled": False,
                "risk_level": "high",
                "parameters": {"model": "xgboost", "confidence_threshold": 0.7}
            }
        ]
    }

@app.get("/api/v1/market-data/{symbol}")
async def get_market_data(symbol: str, limit: int = 100):
    import random
    data = []
    base_price = 50000.0
    
    for i in range(min(limit, 100)):
        change = random.uniform(-0.02, 0.02)
        price = base_price * (1 + change)
        data.append({
            "timestamp": datetime.utcnow().isoformat(),
            "close": price,
            "volume": random.uniform(10, 1000)
        })
        base_price = price
    
    return {
        "symbol": symbol,
        "data": data,
        "latest_price": data[-1]["close"] if data else 0
    }

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 AI Trading Bot Starting...")
    logger.info("=" * 60)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
