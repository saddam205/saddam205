from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import random
import os

app = FastAPI(
    title="AI Trading Bot API",
    description="Advanced AI-powered cryptocurrency trading bot",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
# Mount the current directory to serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Root endpoint to serve dashboard
@app.get("/")
async def root():
    return FileResponse("dashboard.html")

# Dashboard endpoint
@app.get("/dashboard")
@app.get("/dashboard.html")
async def dashboard():
    return FileResponse("dashboard.html")

# API Models
class TradeRequest(BaseModel):
    symbol: str
    quantity: float
    side: str

class TradeResponse(BaseModel):
    status: str
    trade_id: str
    symbol: str
    quantity: float
    side: str
    price: float
    timestamp: str
    message: str

# Health endpoint
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "trading_bot"
    }

# Status endpoint
@app.get("/api/v1/status")
async def get_status():
    return {
        "status": "running",
        "mode": "paper_trading",
        "portfolio_value": 100000.0,
        "active_strategies": ["momentum", "mean_reversion", "trend_following"],
        "daily_pnl": 1250.0,
        "total_trades": 156,
        "win_rate": 62.8
    }

# Trade endpoint
@app.post("/api/v1/trade", response_model=TradeResponse)
async def execute_trade(trade: TradeRequest):
    prices = {
        "BTC/USDT": 50000,
        "ETH/USDT": 3000,
        "SOL/USDT": 100,
        "DOGE/USDT": 0.15
    }
    base_price = prices.get(trade.symbol, 50000)
    price = base_price + random.uniform(-base_price * 0.02, base_price * 0.02)
    
    return TradeResponse(
        status="success",
        trade_id=f"trade_{int(datetime.utcnow().timestamp())}_{random.randint(1000, 9999)}",
        symbol=trade.symbol,
        quantity=trade.quantity,
        side=trade.side,
        price=round(price, 2),
        timestamp=datetime.utcnow().isoformat(),
        message=f"{trade.side.upper()} order for {trade.quantity} {trade.symbol} executed at ${round(price, 2)}"
    )

# Portfolio endpoint
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

# Strategies endpoint
@app.get("/api/v1/strategies")
async def get_strategies():
    return {
        "strategies": [
            {
                "id": 1,
                "name": "momentum",
                "display_name": "Momentum Strategy",
                "description": "Follows market momentum using RSI and MACD",
                "enabled": True,
                "risk_level": "medium",
                "parameters": {"period": 14, "threshold": 2.0},
                "performance": {"win_rate": 65.5, "total_trades": 45, "profit": 12500.0}
            },
            {
                "id": 2,
                "name": "mean_reversion",
                "display_name": "Mean Reversion",
                "description": "Trades based on price mean reversion using Bollinger Bands",
                "enabled": True,
                "risk_level": "low",
                "parameters": {"lookback": 20, "std_dev": 2.0},
                "performance": {"win_rate": 58.2, "total_trades": 38, "profit": 8900.0}
            }
        ]
    }

# Market data endpoint
@app.get("/api/v1/market-data/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "1h", limit: int = 100):
    data = []
    base_price = 50000.0
    
    for i in range(min(limit, 100)):
        change = random.uniform(-0.03, 0.03)
        close = base_price * (1 + change)
        data.append({
            "timestamp": datetime.utcnow().isoformat(),
            "open": base_price,
            "high": max(base_price, close) * (1 + random.uniform(0, 0.01)),
            "low": min(base_price, close) * (1 - random.uniform(0, 0.01)),
            "close": close,
            "volume": random.uniform(100, 10000)
        })
        base_price = close
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "data": data,
        "latest_price": data[-1]["close"] if data else 0,
        "change_24h": random.uniform(-5, 5)
    }

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 AI TRADING BOT BACKEND")
    print("=" * 60)
    print("📍 API URL: http://localhost:8000")
    print("📍 Dashboard: http://localhost:8000/dashboard.html")
    print("📍 API Docs: http://localhost:8000/docs")
    print("📍 Health: http://localhost:8000/health")
    print("=" * 60)
    print("✅ Backend is ready!")
    print("Press CTRL+C to stop")
    print("=" * 60)
    
    uvicorn.run(
        "test_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
