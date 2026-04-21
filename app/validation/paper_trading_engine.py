"""
paper_trading_engine.py
Part of the app/validation module.
Paper trading engine for simulation before live deployment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import logging
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Paper trade record"""
    id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = 'OPEN'
    metadata: Dict[str, Any] = field(default_factory=dict)


class PaperTradingEngine:
    """
    Paper trading engine for strategy validation before live deployment.
    Simulates real trading conditions without risking actual capital.
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 state_file: str = "data/paper_trading/state.json"):
        """
        Initialize paper trading engine
        
        Args:
            initial_capital: Starting virtual capital
            state_file: File to persist trading state
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, PaperTrade] = {}
        self.trades: List[PaperTrade] = []
        self.state_file = state_file
        self.daily_pnl = []
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        
        # Load previous state
        self._load_state()
        
    def execute_trade(self, symbol: str, side: str, quantity: float,
                     price: float, metadata: Dict = None) -> Optional[PaperTrade]:
        """
        Execute a paper trade
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Trade quantity
            price: Execution price
            metadata: Additional trade data
        
        Returns:
            PaperTrade object if successful
        """
        trade_id = f"paper_{datetime.now().timestamp()}_{symbol}"
        
        if side.upper() == 'BUY':
            cost = quantity * price
            if cost > self.capital:
                logger.warning(f"Insufficient paper capital: need ${cost:.2f}, have ${self.capital:.2f}")
                return None
            
            self.capital -= cost
            
            trade = PaperTrade(
                id=trade_id,
                symbol=symbol,
                side='BUY',
                quantity=quantity,
                entry_price=price,
                entry_time=datetime.now(),
                status='OPEN',
                metadata=metadata or {}
            )
            
            self.positions[trade_id] = trade
            logger.info(f"Paper BUY: {quantity} {symbol} @ ${price:.2f}")
            
        elif side.upper() == 'SELL':
            # Find open position
            open_position = None
            for pos_id, pos in self.positions.items():
                if pos.symbol == symbol and pos.status == 'OPEN':
                    open_position = pos
                    break
            
            if not open_position:
                logger.warning(f"No open position found for {symbol}")
                return None
            
            revenue = quantity * price
            self.capital += revenue
            
            # Calculate P&L
            if open_position.side == 'BUY':
                pnl = (price - open_position.entry_price) * quantity
                pnl_pct = (price / open_position.entry_price - 1) * 100
            else:
                pnl = (open_position.entry_price - price) * quantity
                pnl_pct = (open_position.entry_price / price - 1) * 100
            
            open_position.exit_price = price
            open_position.exit_time = datetime.now()
            open_position.pnl = pnl
            open_position.pnl_pct = pnl_pct
            open_position.status = 'CLOSED'
            
            self.trades.append(open_position)
            del self.positions[open_position.id]
            
            # Update daily P&L
            self._update_daily_pnl(pnl)
            
            logger.info(f"Paper SELL: {quantity} {symbol} @ ${price:.2f}, P&L: ${pnl:.2f}")
            return open_position
        
        self._save_state()
        return trade
    
    def _update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        today = datetime.now().date()
        
        if self.daily_pnl and self.daily_pnl[-1]['date'] == today:
            self.daily_pnl[-1]['pnl'] += pnl
        else:
            self.daily_pnl.append({'date': today, 'pnl': pnl})
        
        # Keep last 365 days
        if len(self.daily_pnl) > 365:
            self.daily_pnl = self.daily_pnl[-365:]
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value
        
        Args:
            current_prices: Dictionary of current prices by symbol
        
        Returns:
            Total portfolio value
        """
        total = self.capital
        
        for position in self.positions.values():
            if position.symbol in current_prices:
                price = current_prices[position.symbol]
                if position.side == 'BUY':
                    total += position.quantity * price
                else:
                    total += position.quantity * (2 * position.entry_price - price)
        
        return total
    
    def get_performance(self) -> Dict:
        """Get trading performance metrics"""
        closed_trades = [t for t in self.trades if t.status == 'CLOSED']
        
        if not closed_trades:
            return {
                'initial_capital': self.initial_capital,
                'current_capital': self.capital,
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'open_positions': len(self.positions)
            }
        
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        total_pnl = sum(t.pnl for t in closed_trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.capital,
            'total_return': total_return,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(closed_trades) - len(winning_trades),
            'win_rate': len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(closed_trades) if closed_trades else 0,
            'open_positions': len(self.positions),
            'daily_pnl': self.daily_pnl[-30:]  # Last 30 days
        }
    
    def get_positions(self) -> List[Dict]:
        """Get current open positions"""
        return [
            {
                'id': p.id,
                'symbol': p.symbol,
                'side': p.side,
                'quantity': p.quantity,
                'entry_price': p.entry_price,
                'entry_time': p.entry_time.isoformat(),
                'unrealized_pnl': 0  # Will be calculated with current price
            }
            for p in self.positions.values()
        ]
    
    def reset(self):
        """Reset paper trading account"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_pnl = []
        self._save_state()
        logger.info("Paper trading account reset")
    
    def _save_state(self):
        """Save paper trading state to file"""
        try:
            state = {
                'initial_capital': self.initial_capital,
                'capital': self.capital,
                'positions': [
                    {
                        'id': p.id,
                        'symbol': p.symbol,
                        'side': p.side,
                        'quantity': p.quantity,
                        'entry_price': p.entry_price,
                        'entry_time': p.entry_time.isoformat(),
                        'metadata': p.metadata
                    }
                    for p in self.positions.values()
                ],
                'trades': [
                    {
                        'id': t.id,
                        'symbol': t.symbol,
                        'side': t.side,
                        'quantity': t.quantity,
                        'entry_price': t.entry_price,
                        'exit_price': t.exit_price,
                        'entry_time': t.entry_time.isoformat(),
                        'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                        'pnl': t.pnl,
                        'pnl_pct': t.pnl_pct,
                        'status': t.status
                    }
                    for t in self.trades[-100:]  # Last 100 trades
                ],
                'daily_pnl': self.daily_pnl[-30:]
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save paper trading state: {e}")
    
    def _load_state(self):
        """Load paper trading state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.initial_capital = state.get('initial_capital', 100000)
                self.capital = state.get('capital', self.initial_capital)
                self.daily_pnl = state.get('daily_pnl', [])
                
                # Load positions
                for pos_data in state.get('positions', []):
                    pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])
                    self.positions[pos_data['id']] = PaperTrade(**pos_data)
                
                # Load trades
                for trade_data in state.get('trades', []):
                    trade_data['entry_time'] = datetime.fromisoformat(trade_data['entry_time'])
                    if trade_data.get('exit_time'):
                        trade_data['exit_time'] = datetime.fromisoformat(trade_data['exit_time'])
                    self.trades.append(PaperTrade(**trade_data))
                
                logger.info(f"Loaded paper trading state: ${self.capital:.2f} capital")
                
        except Exception as e:
            logger.error(f"Failed to load paper trading state: {e}")