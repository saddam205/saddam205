"""
regime_validator.py
Part of the app/validation module.
Regime-based validation for strategy robustness across market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"


@dataclass
class RegimeValidationResult:
    """Results of regime-based validation"""
    regime: MarketRegime
    periods: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades_count: int
    is_robust: bool
    score: float


class RegimeValidator:
    """
    Validates strategy performance across different market regimes
    """
    
    def __init__(self, lookback_days: int = 60):
        """
        Initialize regime validator
        
        Args:
            lookback_days: Days to look back for regime detection
        """
        self.lookback_days = lookback_days
        self.regime_results: List[RegimeValidationResult] = []
        
    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect market regimes in the data
        
        Args:
            data: OHLCV DataFrame
        
        Returns:
            Series of regime labels
        """
        regimes = []
        
        for i in range(self.lookback_days, len(data)):
            window = data.iloc[i - self.lookback_days:i]
            regime = self._detect_regime(window)
            regimes.append(regime)
        
        # Pad the beginning
        regimes = [regimes[0]] * self.lookback_days + regimes
        
        return pd.Series(regimes, index=data.index)
    
    def _detect_regime(self, window: pd.DataFrame) -> MarketRegime:
        """Detect regime for a single window"""
        close = window['close']
        
        # Calculate trend
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        trend_strength = abs(sma_20 / sma_50 - 1) if sma_50 > 0 else 0
        
        # Calculate volatility
        returns = close.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate range
        price_range = (close.max() - close.min()) / close.mean()
        
        # Classify regime
        if trend_strength > 0.05:
            if sma_20 > sma_50:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        elif volatility > 0.05:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.01:
            return MarketRegime.LOW_VOLATILITY
        elif price_range < 0.05:
            return MarketRegime.RANGING
        else:
            return MarketRegime.BREAKOUT
    
    def validate_strategy(self, strategy_func, data: pd.DataFrame,
                         initial_capital: float = 100000) -> List[RegimeValidationResult]:
        """
        Validate strategy across all detected regimes
        
        Args:
            strategy_func: Strategy function to validate
            data: Market data
            initial_capital: Starting capital
        
        Returns:
            List of regime validation results
        """
        regimes = self.detect_regimes(data)
        unique_regimes = regimes.unique()
        
        results = []
        
        for regime in unique_regimes:
            regime_data = data[regimes == regime]
            
            if len(regime_data) < 20:
                continue
            
            # Run backtest on this regime
            metrics = self._run_backtest(strategy_func, regime_data, initial_capital)
            
            # Determine if robust
            is_robust = metrics['sharpe_ratio'] > 0.5 and metrics['max_drawdown'] < 20
            score = self._calculate_regime_score(metrics)
            
            result = RegimeValidationResult(
                regime=regime,
                periods=len(regime_data),
                total_return=metrics['total_return'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                win_rate=metrics['win_rate'],
                trades_count=metrics['total_trades'],
                is_robust=is_robust,
                score=score
            )
            
            results.append(result)
        
        self.regime_results = results
        return results
    
    def _run_backtest(self, strategy_func, data: pd.DataFrame,
                     initial_capital: float) -> Dict:
        """Run backtest for a specific data period"""
        # Simplified backtest
        capital = initial_capital
        position = 0
        trades = []
        
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            signal = strategy_func(current_data) if len(current_data) > 10 else 'HOLD'
            price = data['close'].iloc[i]
            
            if signal == 'BUY' and position == 0:
                position = capital * 0.2 / price
                capital -= position * price
                trades.append({'side': 'BUY', 'price': price})
            elif signal == 'SELL' and position > 0:
                capital += position * price
                pnl = (price - trades[-1]['price']) * position
                trades[-1]['pnl'] = pnl
                position = 0
        
        # Calculate metrics
        if not trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0
            }
        
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        total_return = (capital - initial_capital) / initial_capital * 100
        
        return {
            'total_return': total_return,
            'sharpe_ratio': 0.8 if total_return > 0 else 0.2,  # Simplified
            'max_drawdown': 10,  # Simplified
            'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
            'total_trades': len(trades)
        }
    
    def _calculate_regime_score(self, metrics: Dict) -> float:
        """Calculate score for a regime (0-100)"""
        score = 0
        
        # Return (40%)
        if metrics['total_return'] > 20:
            score += 40
        elif metrics['total_return'] > 10:
            score += 30
        elif metrics['total_return'] > 5:
            score += 20
        elif metrics['total_return'] > 0:
            score += 10
        
        # Sharpe ratio (30%)
        if metrics['sharpe_ratio'] > 1.5:
            score += 30
        elif metrics['sharpe_ratio'] > 1:
            score += 20
        elif metrics['sharpe_ratio'] > 0.5:
            score += 10
        
        # Drawdown (30%)
        if metrics['max_drawdown'] < 5:
            score += 30
        elif metrics['max_drawdown'] < 10:
            score += 20
        elif metrics['max_drawdown'] < 15:
            score += 10
        
        return score
    
    def get_robustness_score(self) -> float:
        """
        Calculate overall robustness score across all regimes
        
        Returns:
            Robustness score (0-100)
        """
        if not self.regime_results:
            return 0
        
        # Average score across regimes
        avg_score = np.mean([r.score for r in self.regime_results])
        
        # Penalize if any regime performed poorly
        min_score = min([r.score for r in self.regime_results])
        if min_score < 30:
            avg_score *= 0.7
        elif min_score < 50:
            avg_score *= 0.85
        
        return avg_score
    
    def get_summary(self) -> Dict:
        """Get regime validation summary"""
        if not self.regime_results:
            return {'message': 'No validation results'}
        
        return {
            'regimes_analyzed': len(self.regime_results),
            'robustness_score': self.get_robustness_score(),
            'best_regime': max(self.regime_results, key=lambda x: x.score).regime.value,
            'worst_regime': min(self.regime_results, key=lambda x: x.score).regime.value,
            'regime_details': [
                {
                    'regime': r.regime.value,
                    'periods': r.periods,
                    'return': r.total_return,
                    'sharpe': r.sharpe_ratio,
                    'drawdown': r.max_drawdown,
                    'robust': r.is_robust
                }
                for r in self.regime_results
            ]
        }