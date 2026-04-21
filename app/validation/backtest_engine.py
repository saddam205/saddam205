"""
backtest_engine.py
Part of the app/validation module.
Backtest engine with realistic costs and slippage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    pnl: float
    pnl_pct: float
    fees: float
    slippage: float


class RealisticCostCalculator:
    """
    Professional cost modeling for realistic P&L
    """
    
    def __init__(self, 
                 maker_fee: float = 0.0005,
                 taker_fee: float = 0.001,
                 slippage_model: str = 'adaptive',
                 spread_model: str = 'volatility_based'):
        
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_model = slippage_model
        self.spread_model = spread_model
        self.slippage_history = []
        
    def calculate_trade_cost(self, 
                            trade_size: float,
                            price: float,
                            volatility: float,
                            volume: float,
                            is_market_order: bool = True) -> Dict:
        """
        Calculate realistic trade costs including:
        - Exchange fees
        - Slippage
        - Spread
        - Market impact
        """
        # 1. Exchange Fee
        fee_rate = self.taker_fee if is_market_order else self.maker_fee
        exchange_fee = trade_size * price * fee_rate
        
        # 2. Slippage
        slippage = self._calculate_slippage(trade_size, volume, volatility, is_market_order)
        slippage_cost = trade_size * price * slippage
        
        # 3. Spread Cost
        spread = self._calculate_spread(volatility, volume)
        spread_cost = trade_size * price * (spread / 2)
        
        # 4. Market Impact
        market_impact = self._calculate_market_impact(trade_size, volume, price)
        
        total_cost = exchange_fee + slippage_cost + spread_cost + market_impact
        total_cost_pct = total_cost / (trade_size * price) * 100 if trade_size * price > 0 else 0
        
        return {
            'exchange_fee': exchange_fee,
            'exchange_fee_pct': fee_rate * 100,
            'slippage': slippage,
            'slippage_cost': slippage_cost,
            'slippage_pct': slippage * 100,
            'spread': spread,
            'spread_cost': spread_cost,
            'spread_pct': (spread / 2) * 100,
            'market_impact': market_impact,
            'market_impact_pct': (market_impact / (trade_size * price)) * 100 if trade_size * price > 0 else 0,
            'total_cost': total_cost,
            'total_cost_pct': total_cost_pct
        }
    
    def _calculate_slippage(self, trade_size: float, volume: float, 
                           volatility: float, is_market_order: bool) -> float:
        """Calculate expected slippage"""
        if is_market_order:
            base_slippage = 0.0005
        else:
            base_slippage = 0.0001
        
        size_ratio = trade_size / volume if volume > 0 else 0.01
        size_impact = min(size_ratio * 10, 0.005)
        
        vol_impact = volatility * 0.5
        
        total_slippage = base_slippage + size_impact + vol_impact
        
        self.slippage_history.append({
            'trade_size': trade_size,
            'volume': volume,
            'volatility': volatility,
            'slippage': total_slippage
        })
        
        return min(total_slippage, 0.01)
    
    def _calculate_spread(self, volatility: float, volume: float) -> float:
        """Calculate bid-ask spread"""
        base_spread = 0.0005
        vol_adjustment = volatility * 0.5
        liquidity_score = min(volume / 1000000, 1)
        liquidity_adjustment = (1 - liquidity_score) * 0.001
        
        total_spread = base_spread + vol_adjustment + liquidity_adjustment
        return min(total_spread, 0.005)
    
    def _calculate_market_impact(self, trade_size: float, volume: float, price: float) -> float:
        """Calculate market impact for large orders"""
        if volume == 0:
            return 0
        
        participation = trade_size / volume
        eta = 0.1
        sigma = 0.02
        
        impact_pct = eta * sigma * np.sqrt(participation)
        impact_pct = min(impact_pct, 0.01)
        
        return trade_size * price * impact_pct


class BacktestEngine:
    """
    Backtest engine with realistic costs and comprehensive metrics
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize backtest engine
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.cost_calculator = RealisticCostCalculator()
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = [initial_capital]
        
    def execute_trade(self, side: str, size: float, price: float,
                     timestamp: datetime, volatility: float, volume: float) -> Optional[TradeRecord]:
        """
        Execute a trade with realistic costs
        
        Args:
            side: 'BUY' or 'SELL'
            size: Position size
            price: Execution price
            timestamp: Trade timestamp
            volatility: Current volatility
            volume: Trading volume
        
        Returns:
            TradeRecord if successful
        """
        # Calculate costs
        costs = self.cost_calculator.calculate_trade_cost(
            trade_size=size,
            price=price,
            volatility=volatility,
            volume=volume,
            is_market_order=True
        )
        
        # Execute trade
        if side.upper() == 'BUY':
            total_cost = size * price + costs['total_cost']
            if total_cost > self.capital:
                logger.warning(f"Insufficient capital: need ${total_cost:.2f}, have ${self.capital:.2f}")
                return None
            self.capital -= total_cost
            position_size = size
            entry_price = price
        else:  # SELL
            net_proceeds = size * price - costs['total_cost']
            self.capital += net_proceeds
            position_size = -size
            entry_price = price
        
        trade = TradeRecord(
            entry_time=timestamp,
            exit_time=timestamp,
            entry_price=entry_price,
            exit_price=price,
            quantity=size,
            side=side.upper(),
            pnl=0,  # Will be calculated on close
            pnl_pct=0,
            fees=costs['exchange_fee'],
            slippage=costs['slippage_cost']
        )
        
        return trade
    
    def close_position(self, position: TradeRecord, exit_price: float,
                      timestamp: datetime, volatility: float, volume: float) -> TradeRecord:
        """Close an existing position"""
        # Calculate costs for closing
        costs = self.cost_calculator.calculate_trade_cost(
            trade_size=position.quantity,
            price=exit_price,
            volatility=volatility,
            volume=volume,
            is_market_order=True
        )
        
        # Calculate P&L
        if position.side == 'BUY':
            gross_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - exit_price) * position.quantity
        
        total_costs = position.fees + position.slippage + costs['total_cost']
        net_pnl = gross_pnl - total_costs
        net_pnl_pct = (net_pnl / (position.entry_price * position.quantity)) * 100 if position.entry_price * position.quantity > 0 else 0
        
        # Update trade record
        position.exit_time = timestamp
        position.exit_price = exit_price
        position.pnl = net_pnl
        position.pnl_pct = net_pnl_pct
        position.fees += costs['exchange_fee']
        position.slippage += costs['slippage_cost']
        
        # Update capital
        self.capital += position.quantity * exit_price - costs['total_cost']
        
        return position
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {'message': 'No trades executed'}
        
        closed_trades = [t for t in self.trades if t.exit_time != t.entry_time]
        
        if not closed_trades:
            return {'message': 'No closed trades'}
        
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in closed_trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        returns = [t.pnl_pct for t in closed_trades]
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        
        # Calculate Sharpe ratio from equity curve
        equity_returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe_ratio = np.mean(equity_returns) / (np.std(equity_returns) + 1e-8) * np.sqrt(252)
        
        # Calculate max drawdown
        peak = self.equity_curve[0]
        max_drawdown = 0
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_drawdown = max(max_drawdown, dd)
        
        total_costs = sum(t.fees + t.slippage for t in self.trades)
        cost_drag = total_costs / self.initial_capital * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'total_costs': total_costs,
            'cost_drag_pct': cost_drag,
            'avg_trade_pnl': total_pnl / len(closed_trades) if closed_trades else 0,
            'best_trade': max(t.pnl for t in closed_trades) if closed_trades else 0,
            'worst_trade': min(t.pnl for t in closed_trades) if closed_trades else 0
        }


class CostAdjustedBacktest(BacktestEngine):
    """Backtest engine with enhanced cost modeling"""
    
    def __init__(self, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.cost_calculator = RealisticCostCalculator()
    
    def get_realistic_performance(self) -> Dict:
        """Get performance metrics with costs included"""
        metrics = self.get_performance_metrics()
        
        # Add cost-specific metrics
        total_fees = sum(t.fees for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)
        
        metrics['total_fees'] = total_fees
        metrics['total_slippage'] = total_slippage
        metrics['avg_fee_per_trade'] = total_fees / len(self.trades) if self.trades else 0
        metrics['avg_slippage_per_trade'] = total_slippage / len(self.trades) if self.trades else 0
        
        return metrics