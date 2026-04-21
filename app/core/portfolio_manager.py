"""
portfolio_manager.py
Part of the app/core module.
Portfolio tracking and management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Trading position data"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    side: str  # 'LONG' or 'SHORT'
    current_price: float = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: float = None
    take_profit: float = None
    
    def update_pnl(self, current_price: float) -> float:
        """Update unrealized P&L"""
        self.current_price = current_price
        
        if self.side == 'LONG':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
        
        return self.unrealized_pnl
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat()
        return data


@dataclass
class Trade:
    """Completed trade record"""
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    fees: float
    confidence: float
    strategy: str = None


class PortfolioManager:
    """
    Manages portfolio positions, P&L tracking, and performance metrics
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 state_file: str = "data/portfolio_state.json"):
        """
        Initialize portfolio manager
        
        Args:
            initial_capital: Starting capital
            state_file: File to persist portfolio state
        """
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.state_file = state_file
        self.daily_pnl = []
        self.equity_curve = []
        self.websocket_subscribers = {}
        
        # Performance tracking
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Load previous state
        self._load_state()
        
    def add_position(self, symbol: str, quantity: float, price: float, 
                     side: str, stop_loss: float = None, 
                     take_profit: float = None) -> Position:
        """
        Add a new position
        
        Args:
            symbol: Asset symbol
            quantity: Position quantity
            price: Entry price
            side: 'LONG' or 'SHORT'
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Position object
        """
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}, closing existing")
            self.close_position(symbol, price)
        
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Deduct cash
        position_value = quantity * price
        self.cash_balance -= position_value
        
        self.positions[symbol] = position
        self._save_state()
        
        logger.info(f"Opened {side} position: {quantity} {symbol} @ ${price:.2f}")
        
        return position
    
    def close_position(self, symbol: str, price: float, 
                       reason: str = "manual") -> Optional[Trade]:
        """
        Close an existing position
        
        Args:
            symbol: Asset symbol
            price: Exit price
            reason: Reason for closing
        
        Returns:
            Trade record if successful
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.side == 'LONG':
            pnl = (price - position.entry_price) * position.quantity
            pnl_percent = (price / position.entry_price - 1) * 100
        else:  # SHORT
            pnl = (position.entry_price - price) * position.quantity
            pnl_percent = (position.entry_price / price - 1) * 100
        
        # Apply fees (0.1% assumed)
        fees = price * position.quantity * 0.001
        pnl -= fees
        
        # Update cash balance
        position_value = price * position.quantity
        self.cash_balance += position_value
        
        # Create trade record
        trade = Trade(
            id=f"{symbol}_{datetime.now().timestamp()}",
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            pnl_percent=pnl_percent,
            fees=fees,
            confidence=1.0,
            strategy=reason
        )
        
        self.trades.append(trade)
        
        # Update daily P&L
        self._update_daily_pnl(pnl)
        
        # Remove position
        del self.positions[symbol]
        self._save_state()
        
        logger.info(f"Closed {position.side} position: {position.quantity} {symbol} @ ${price:.2f}")
        logger.info(f"P&L: ${pnl:.2f} ({pnl_percent:.2f}%)")
        
        return trade
    
    def update_positions(self, current_prices: Dict[str, float]) -> Dict[str, Dict]:
        """
        Update all positions with current prices
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
        
        Returns:
            Updated position data
        """
        updated_positions = {}
        total_equity = self.cash_balance
        total_unrealized_pnl = 0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                price = current_prices[symbol]
                unrealized_pnl = position.update_pnl(price)
                total_unrealized_pnl += unrealized_pnl
                
                # Check stop loss and take profit
                if position.stop_loss and self._check_stop_loss(position, price):
                    self.close_position(symbol, price, reason="stop_loss")
                    continue
                    
                if position.take_profit and self._check_take_profit(position, price):
                    self.close_position(symbol, price, reason="take_profit")
                    continue
                
                position_value = position.quantity * price
                total_equity += position_value
                
                updated_positions[symbol] = {
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': price,
                    'unrealized_pnl': unrealized_pnl,
                    'value': position_value,
                    'side': position.side
                }
        
        # Update drawdown
        total_equity += total_unrealized_pnl
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
        
        self.current_drawdown = (self.peak_equity - total_equity) / self.peak_equity * 100
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Update equity curve
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': total_equity,
            'cash': self.cash_balance,
            'unrealized_pnl': total_unrealized_pnl,
            'drawdown': self.current_drawdown
        })
        
        # Keep only last 1000 records
        if len(self.equity_curve) > 1000:
            self.equity_curve = self.equity_curve[-1000:]
        
        self._save_state()
        
        return updated_positions
    
    def _check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop loss is triggered"""
        if position.side == 'LONG':
            return current_price <= position.stop_loss
        else:  # SHORT
            return current_price >= position.stop_loss
    
    def _check_take_profit(self, position: Position, current_price: float) -> bool:
        """Check if take profit is triggered"""
        if position.side == 'LONG':
            return current_price >= position.take_profit
        else:  # SHORT
            return current_price <= position.take_profit
    
    def _update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        today = datetime.now().date()
        
        # Find or create today's record
        for record in self.daily_pnl:
            if record['date'] == today:
                record['pnl'] += pnl
                break
        else:
            self.daily_pnl.append({
                'date': today,
                'pnl': pnl
            })
        
        # Keep last 365 days
        if len(self.daily_pnl) > 365:
            self.daily_pnl = self.daily_pnl[-365:]
    
    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        total = self.cash_balance
        
        for position in self.positions.values():
            if position.current_price:
                total += position.quantity * position.current_price
        
        return total
    
    def get_cash_balance(self) -> float:
        """Get current cash balance"""
        return self.cash_balance
    
    def get_positions(self) -> List[Dict]:
        """Get all current positions"""
        return [p.to_dict() for p in self.positions.values()]
    
    def get_open_trades(self) -> List[Dict]:
        """Get all open trades"""
        return [p.to_dict() for p in self.positions.values()]
    
    def get_daily_pnl(self) -> List[Dict]:
        """Get daily P&L history"""
        return self.daily_pnl[-30:]  # Last 30 days
    
    def get_total_pnl(self) -> float:
        """Get total realized P&L"""
        return sum(trade.pnl for trade in self.trades)
    
    def get_total_return(self) -> float:
        """Get total return percentage"""
        current_value = self.get_total_value()
        return (current_value - self.initial_capital) / self.initial_capital * 100
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        total_value = self.get_total_value()
        total_return = self.get_total_return()
        
        # Calculate Sharpe ratio if enough data
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                ret = (self.equity_curve[i]['equity'] - self.equity_curve[i-1]['equity']) / self.equity_curve[i-1]['equity']
                returns.append(ret)
            
            returns = np.array(returns)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate win rate
        winning_trades = [t for t in self.trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_value': total_value,
            'cash_balance': self.cash_balance,
            'total_return': total_return,
            'total_pnl': self.get_total_pnl(),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'open_positions': len(self.positions),
            'peak_equity': self.peak_equity,
            'daily_pnl': self.get_daily_pnl()
        }
    
    def get_performance_report(self) -> str:
        """Generate human-readable performance report"""
        metrics = self.get_performance_metrics()
        
        report = []
        report.append("=" * 50)
        report.append("PORTFOLIO PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"Total Value: ${metrics['total_value']:,.2f}")
        report.append(f"Cash Balance: ${metrics['cash_balance']:,.2f}")
        report.append(f"Total Return: {metrics['total_return']:.2f}%")
        report.append(f"Total P&L: ${metrics['total_pnl']:,.2f}")
        report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        report.append(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        report.append(f"Win Rate: {metrics['win_rate']:.1f}%")
        report.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
        report.append(f"Total Trades: {metrics['total_trades']}")
        report.append(f"Open Positions: {metrics['open_positions']}")
        report.append("=" * 50)
        
        return "\n".join(report)
    
    def record_trade(self, order: Dict, signal: Dict):
        """Record a trade after execution"""
        trade = Trade(
            id=order.get('id', f"trade_{datetime.now().timestamp()}"),
            symbol=order['symbol'],
            side=signal['signal'],
            entry_price=order['price'],
            exit_price=None,  # Will be set when closed
            quantity=order['quantity'],
            entry_time=datetime.now(),
            exit_time=None,
            pnl=0,
            pnl_percent=0,
            fees=order.get('fees', 0),
            confidence=signal['confidence']
        )
        
        self.trades.append(trade)
        self._save_state()
    
    def subscribe_websocket(self, symbol: str, websocket):
        """Subscribe a websocket to symbol updates"""
        if symbol not in self.websocket_subscribers:
            self.websocket_subscribers[symbol] = []
        self.websocket_subscribers[symbol].append(websocket)
    
    def _save_state(self):
        """Persist portfolio state to file"""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            state = {
                'cash_balance': self.cash_balance,
                'positions': {s: p.to_dict() for s, p in self.positions.items()},
                'trades': [asdict(t) for t in self.trades[-100:]],  # Last 100 trades
                'daily_pnl': self.daily_pnl[-30:],  # Last 30 days
                'peak_equity': self.peak_equity,
                'max_drawdown': self.max_drawdown
            }
            
            # Convert datetime objects to strings
            for trade in state['trades']:
                if 'entry_time' in trade and isinstance(trade['entry_time'], datetime):
                    trade['entry_time'] = trade['entry_time'].isoformat()
                if 'exit_time' in trade and trade['exit_time'] and isinstance(trade['exit_time'], datetime):
                    trade['exit_time'] = trade['exit_time'].isoformat()
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save portfolio state: {e}")
    
    def _load_state(self):
        """Load portfolio state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.cash_balance = state.get('cash_balance', self.initial_capital)
                self.peak_equity = state.get('peak_equity', self.initial_capital)
                self.max_drawdown = state.get('max_drawdown', 0.0)
                self.daily_pnl = state.get('daily_pnl', [])
                
                # Load positions
                positions_data = state.get('positions', {})
                for symbol, pos_data in positions_data.items():
                    pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])
                    self.positions[symbol] = Position(**pos_data)
                
                # Load trades
                trades_data = state.get('trades', [])
                for trade_data in trades_data:
                    trade_data['entry_time'] = datetime.fromisoformat(trade_data['entry_time'])
                    if trade_data.get('exit_time'):
                        trade_data['exit_time'] = datetime.fromisoformat(trade_data['exit_time'])
                    self.trades.append(Trade(**trade_data))
                
                logger.info(f"Loaded portfolio state: ${self.cash_balance:,.2f} cash, {len(self.positions)} positions")
                
        except Exception as e:
            logger.error(f"Failed to load portfolio state: {e}")
    
    def reset(self) -> Dict:
        """Reset portfolio to initial state"""
        self.cash_balance = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_pnl = []
        self.equity_curve = []
        self.peak_equity = self.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self._save_state()
        
        logger.info("Portfolio reset to initial state")
        
        return {
            'success': True,
            'message': 'Portfolio reset successfully',
            'initial_capital': self.initial_capital
        }