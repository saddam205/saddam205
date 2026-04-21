import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    timestamp: datetime
    side: OrderSide
    price: float
    quantity: float
    order_type: OrderType = OrderType.MARKET

@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: OrderSide
    pnl: float
    pnl_pct: float

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, data: pd.DataFrame, strategy_func, **strategy_params):
        """
        Run backtest with given strategy
        
        Args:
            data: OHLCV data
            strategy_func: Function that generates signals
            **strategy_params: Parameters for the strategy
        """
        logger.info(f"Starting backtest with {len(data)} bars")
        
        # Generate signals
        signals = strategy_func(data, **strategy_params)
        
        # Simulate trading
        for i in range(len(data)):
            current_bar = data.iloc[i]
            current_time = current_bar.name
            current_price = current_bar['Close']
            signal = signals.iloc[i]
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': self.capital,
                'price': current_price
            })
            
            # Execute signals
            if signal == 1 and not self.positions:  # BUY signal
                self._open_position(current_time, current_price, OrderSide.BUY)
                
            elif signal == -1 and self.positions:  # SELL signal
                self._close_position(current_time, current_price, OrderSide.SELL)
        
        # Close any open positions at the end
        if self.positions:
            self._close_position(data.index[-1], data['Close'].iloc[-1], OrderSide.SELL)
        
        # Calculate performance metrics
        metrics = self.calculate_metrics()
        
        logger.info(f"Backtest completed: {len(self.trades)} trades, Return: {metrics['total_return']:.2f}%")
        
        return metrics, self.trades, self.equity_curve
    
    def _open_position(self, timestamp, price, side):
        """Open a new position"""
        position_size = self.capital * 0.2  # Use 20% of capital per trade
        quantity = position_size / price
        
        order = Order(
            timestamp=timestamp,
            side=side,
            price=price,
            quantity=quantity
        )
        
        self.positions.append(order)
        self.capital -= position_size * (1 + self.commission)
        
    def _close_position(self, timestamp, price, side):
        """Close existing position"""
        if not self.positions:
            return
            
        position = self.positions.pop()
        
        # Calculate P&L
        if position.side == OrderSide.BUY:
            pnl = (price - position.price) * position.quantity
            pnl_pct = (price / position.price - 1) * 100
        else:
            pnl = (position.price - price) * position.quantity
            pnl_pct = (position.price / price - 1) * 100
        
        # Apply commission
        pnl -= (position.price * position.quantity * self.commission)
        
        trade = Trade(
            entry_time=position.timestamp,
            exit_time=timestamp,
            entry_price=position.price,
            exit_price=price,
            quantity=position.quantity,
            side=position.side,
            pnl=pnl,
            pnl_pct=pnl_pct
        )
        
        self.trades.append(trade)
        self.capital += (price * position.quantity) * (1 - self.commission)
        
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # Advanced metrics
        returns = [t.pnl_pct for t in self.trades]
        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / 
                           sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        
        # Sharpe Ratio
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max Drawdown
        equity_values = [e['equity'] for e in self.equity_curve]
        peak = equity_values[0]
        drawdowns = []
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
        max_drawdown = max(drawdowns) * 100
        
        # Calmar Ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'best_trade': max([t.pnl for t in self.trades]),
            'worst_trade': min([t.pnl for t in self.trades])
        }
