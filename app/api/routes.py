"""
API Routes for Trading Bot
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import random
import numpy as np

router = APIRouter()

# Trading state
mock_balance = 10000.0
mock_initial_balance = 10000.0
mock_positions = {}
mock_trades = []

# ============ Balance Endpoints ============

@router.get("/balance")
async def get_balance():
    """Get account balance"""
    global mock_balance
    return {
        "balance": mock_balance,
        "total_pnl": mock_balance - mock_initial_balance,
        "positions": len(mock_positions),
        "available": mock_balance
    }

# ============ Portfolio Endpoints ============

@router.get("/portfolio")
async def get_portfolio():
    """Get portfolio information"""
    global mock_balance, mock_positions
    total_pnl = mock_balance - mock_initial_balance
    
    return {
        "balance": round(mock_balance, 2),
        "total_value": round(mock_balance, 2),
        "total_pnl": round(total_pnl, 2),
        "pnl_percent": round((total_pnl / mock_initial_balance) * 100, 2) if mock_initial_balance > 0 else 0,
        "positions_count": len(mock_positions),
        "positions": mock_positions,
        "initial_balance": mock_initial_balance
    }

@router.get("/positions")
async def get_positions():
    """Get open positions"""
    return mock_positions

# ============ Market Endpoints ============

@router.get("/market/prices")
async def get_market_prices():
    """Get current market prices"""
    symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]
    prices = {}
    for symbol in symbols:
        base = {"BTC-USD": 50000, "ETH-USD": 3000, "BNB-USD": 400, "SOL-USD": 100, "ADA-USD": 0.5}.get(symbol, 100)
        prices[symbol] = round(base * (1 + random.uniform(-0.03, 0.03)), 2)
    return prices

# ============ Performance Endpoints ============

@router.get("/performance")
async def get_performance():
    """Get performance metrics"""
    global mock_trades
    
    total_trades = len(mock_trades)
    winning_trades = len([t for t in mock_trades if t.get("pnl", 0) > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": total_trades - winning_trades,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(mock_balance - mock_initial_balance, 2),
        "sharpe_ratio": round(random.uniform(1.5, 2.2), 2),
        "max_drawdown": round(random.uniform(3, 6), 2)
    }

# ============ Trading Endpoints ============

@router.post("/trade/execute")
async def execute_trade(symbol: str, side: str, quantity: float):
    """Execute a trade"""
    global mock_balance, mock_positions, mock_trades
    
    # Get current price
    price_map = {"BTC-USD": 50000, "ETH-USD": 3000, "BNB-USD": 400, "SOL-USD": 100, "ADA-USD": 0.5}
    base_price = price_map.get(symbol, 100)
    price = base_price * (1 + random.uniform(-0.02, 0.02))
    
    if side.upper() == "BUY":
        cost = quantity * price
        if cost <= mock_balance:
            mock_balance -= cost
            if symbol in mock_positions:
                mock_positions[symbol]["quantity"] += quantity
            else:
                mock_positions[symbol] = {
                    "quantity": quantity,
                    "entry_price": price,
                    "current_price": price
                }
            
            mock_trades.append({
                "id": len(mock_trades) + 1,
                "symbol": symbol,
                "side": "BUY",
                "quantity": quantity,
                "price": round(price, 2),
                "timestamp": datetime.now().isoformat(),
                "pnl": 0
            })
            
            return {
                "success": True,
                "message": f"✅ Bought {quantity} {symbol} at ${price:,.2f}",
                "balance": round(mock_balance, 2)
            }
        else:
            return {
                "success": False,
                "message": f"❌ Insufficient balance. Need ${cost:,.2f}, have ${mock_balance:,.2f}"
            }
    
    elif side.upper() == "SELL":
        if symbol in mock_positions and mock_positions[symbol]["quantity"] >= quantity:
            position = mock_positions[symbol]
            pnl = (price - position["entry_price"]) * quantity
            mock_balance += quantity * price
            
            if mock_positions[symbol]["quantity"] == quantity:
                del mock_positions[symbol]
            else:
                mock_positions[symbol]["quantity"] -= quantity
            
            mock_trades.append({
                "id": len(mock_trades) + 1,
                "symbol": symbol,
                "side": "SELL",
                "quantity": quantity,
                "price": round(price, 2),
                "timestamp": datetime.now().isoformat(),
                "pnl": round(pnl, 2)
            })
            
            return {
                "success": True,
                "message": f"✅ Sold {quantity} {symbol} at ${price:,.2f}, PnL: ${pnl:+,.2f}",
                "balance": round(mock_balance, 2),
                "pnl": round(pnl, 2)
            }
        else:
            return {
                "success": False,
                "message": f"❌ Insufficient position. Have {mock_positions.get(symbol, {}).get('quantity', 0)} {symbol}"
            }
    
    return {"success": False, "message": "Invalid side. Use BUY or SELL"}

# ============ Strategies Endpoints ============

@router.get("/strategies")
async def get_strategies():
    """Get all AI strategies"""
    return {
        "strategies": [
            {"name": "xgboost", "display_name": "XGBoost", "enabled": True, 
             "description": "Gradient boosting for price prediction", 
             "performance": {"win_rate": 62, "profit": 1250}},
            {"name": "bayesian", "display_name": "Bayesian NN", "enabled": True,
             "description": "Neural network with uncertainty estimation",
             "performance": {"win_rate": 58, "profit": 980}},
            {"name": "rl", "display_name": "Reinforcement Learning", "enabled": True,
             "description": "Deep Q-learning for optimal trading",
             "performance": {"win_rate": 65, "profit": 2100}}
        ]
    }

# ============ Status Endpoint ============

@router.get("/status")
async def get_status():
    """Get system status"""
    return {
        "status": "running",
        "mode": "VIRTUAL",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }
# Add this at the top of routes.py
import math

# Global state to track "ticks"
tick_count = 0

@router.get("/market/prices")
async def get_market_prices():
    """Get current market prices with a trend simulation"""
    global tick_count
    tick_count += 1
    
    symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]
    base_prices = {"BTC-USD": 50000, "ETH-USD": 3000, "BNB-USD": 400, "SOL-USD": 100, "ADA-USD": 0.5}
    
    prices = {}
    for symbol, base in base_prices.items():
        # Create a "Wave" effect so prices actually trend up and down
        wave = math.sin(tick_count * 0.1) * 0.02 
        noise = random.uniform(-0.005, 0.005)
        prices[symbol] = round(base * (1 + wave + noise), 2)
    return prices

@router.get("/balance")
async def get_balance():
    """Get account balance that fluctuates with the 'market'"""
    global mock_balance, tick_count
    
    # Simulate PnL changes based on market movement
    simulated_pnl = math.sin(tick_count * 0.05) * 50
    current_display_balance = mock_balance + simulated_pnl
    
    return {
        "balance": round(current_display_balance, 2),
        "total_pnl": round(current_display_balance - mock_initial_balance, 2),
        "positions": len(mock_positions),
        "available": round(current_display_balance, 2),
        "timestamp": datetime.now().isoformat() # Crucial for frontend refresh
    }