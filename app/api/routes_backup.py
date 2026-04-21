"""
API Routes for Trading Bot
"""
from fastapi import APIRouter, HTTPException, WebSocket
from typing import Dict, List, Optional
from datetime import datetime
import random

router = APIRouter(prefix="/api/v1", tags=["trading"])

# Mock data for now
mock_balance = 10000.0
mock_positions = {}

@router.get("/balance")
async def get_balance():
    """Get account balance"""
    return {
        "balance": mock_balance,
        "total_pnl": mock_balance - 10000,
        "positions": len(mock_positions)
    }

@router.get("/positions")
async def get_positions():
    """Get open positions"""
    return mock_positions

@router.get("/performance")
async def get_performance():
    """Get performance metrics"""
    return {
        "total_trades": random.randint(10, 100),
        "winning_trades": random.randint(5, 60),
        "win_rate": round(random.uniform(45, 75), 2),
        "total_pnl": round(random.uniform(-500, 1500), 2),
        "sharpe_ratio": round(random.uniform(1.2, 2.5), 2),
        "max_drawdown": round(random.uniform(2, 8), 2)
    }

@router.post("/trade/execute")
async def execute_trade(symbol: str, side: str, quantity: float):
    """Execute a trade"""
    global mock_balance, mock_positions
    
    price = random.uniform(45000, 55000) if "BTC" in symbol else random.uniform(2800, 3200)
    
    if side.upper() == "BUY":
        cost = quantity * price
        if cost <= mock_balance:
            mock_balance -= cost
            mock_positions[symbol] = {
                "quantity": quantity,
                "entry_price": price,
                "current_price": price
            }
            return {"success": True, "message": f"Bought {quantity} {symbol} at ${price:.2f}", "balance": mock_balance}
        else:
            return {"success": False, "message": "Insufficient balance"}
    else:  # SELL
        if symbol in mock_positions:
            position = mock_positions[symbol]
            pnl = (price - position["entry_price"]) * quantity
            mock_balance += quantity * price
            del mock_positions[symbol]
            return {"success": True, "message": f"Sold {quantity} {symbol} at ${price:.2f}, PnL: ${pnl:.2f}", "balance": mock_balance}
        else:
            return {"success": False, "message": "No position to sell"}

@router.get("/market/prices")
async def get_market_prices():
    """Get current market prices"""
    return {
        "BTC-USD": round(random.uniform(45000, 55000), 2),
        "ETH-USD": round(random.uniform(2800, 3200), 2),
        "BNB-USD": round(random.uniform(380, 420), 2),
        "SOL-USD": round(random.uniform(90, 110), 2),
        "ADA-USD": round(random.uniform(0.45, 0.55), 2)
    }

@router.get("/analysis/sentiment/{symbol}")
async def get_sentiment(symbol: str):
    """Get sentiment analysis"""
    return {
        "symbol": symbol,
        "sentiment_score": random.uniform(-1, 1),
        "sentiment_class": random.choice(["bullish", "bearish", "neutral"]),
        "confidence": random.uniform(0.6, 0.9),
        "timestamp": datetime.now().isoformat()
    }
