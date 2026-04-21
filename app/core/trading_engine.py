"""
Main Trading Engine
"""
import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
from app.core.order_executor import OrderExecutor
from app.config import settings

logger = logging.getLogger(__name__)

class TradingEngine:
    """Core trading engine that manages all trading operations"""
    
    def __init__(self):
        self.order_executor = OrderExecutor()
        self.balance = 10000.0
        self.initial_balance = 10000.0
        self.positions = {}
        self.running = False
        self.trading_mode = settings.TRADING_MODE
        
    async def start(self):
        """Start the trading engine"""
        self.running = True
        logger.info(f"Trading engine started in {self.trading_mode} mode")
        
    async def stop(self):
        """Stop the trading engine"""
        self.running = False
        logger.info("Trading engine stopped")
        
    async def execute_trade(self, symbol: str, side: str, quantity: float, 
                           price: Optional[float] = None) -> Dict:
        """Execute a trade"""
        if not self.running:
            return {'error': 'Trading engine not running'}
        
        # Check balance for buy orders
        if side.upper() == "BUY":
            cost = quantity * (price or 0)
            if cost > self.balance:
                return {'error': f'Insufficient balance. Need ${cost:.2f}, have ${self.balance:.2f}'}
        
        # Execute order
        order = self.order_executor.execute_order(symbol, side, quantity, "MARKET", price)
        
        if order['status'] == 'FILLED':
            if side.upper() == "BUY":
                self.balance -= quantity * order.get('executed_price', price or 0)
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': order.get('executed_price', price or 0),
                    'current_price': order.get('executed_price', price or 0)
                }
            else:  # SELL
                if symbol in self.positions:
                    position = self.positions[symbol]
                    pnl = (order.get('executed_price', price or 0) - position['entry_price']) * quantity
                    self.balance += quantity * order.get('executed_price', price or 0)
                    del self.positions[symbol]
                    order['pnl'] = pnl
        
        return order
    
    def get_balance(self) -> Dict:
        """Get current balance"""
        total_value = self.balance
        for symbol, position in self.positions.items():
            total_value += position['quantity'] * position.get('current_price', position['entry_price'])
        
        return {
            'balance': self.balance,
            'total_value': total_value,
            'total_pnl': total_value - self.initial_balance,
            'positions': self.positions
        }
    
    def get_performance(self) -> Dict:
        """Get performance metrics"""
        total_pnl = self.balance - self.initial_balance
        return {
            'total_pnl': total_pnl,
            'total_value': self.balance,
            'total_trades': len(self.order_executor.orders),
            'open_positions': len(self.positions)
        }

# Global instance
trading_engine = TradingEngine()
