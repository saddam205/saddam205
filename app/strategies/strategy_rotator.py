"""
strategy_rotator.py
Part of the app/strategies module.
Dynamic strategy rotation based on market conditions and performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import logging

from .base_strategy import BaseStrategy, StrategySignal, SignalType
from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy

logger = logging.getLogger(__name__)


class RotationCriteria(Enum):
    """Criteria for strategy rotation"""
    MARKET_REGIME = "market_regime"
    VOLATILITY = "volatility"
    PERFORMANCE = "performance"
    TIME_OF_DAY = "time_of_day"
    CUSTOM = "custom"


class StrategyRotator:
    """
    Dynamic strategy rotator that selects the best strategy based on market conditions.
    Uses performance tracking and market regime detection for optimal selection.
    """
    
    def __init__(self, rotation_interval_minutes: int = 60):
        """
        Initialize strategy rotator
        
        Args:
            rotation_interval_minutes: How often to re-evaluate strategy selection
        """
        self.rotation_interval = rotation_interval_minutes
        self.strategies: Dict[str, BaseStrategy] = {}
        self.current_strategy: Optional[str] = None
        self.last_rotation: Optional[datetime] = None
        self.performance_weights: Dict[str, float] = {}
        self.rotation_history: List[Dict] = []
        
        # Register default strategies
        self._register_default_strategies()
        
    def _register_default_strategies(self):
        """Register default trading strategies"""
        self.strategies['trend_following'] = TrendFollowingStrategy()
        self.strategies['mean_reversion'] = MeanReversionStrategy()
        self.strategies['momentum'] = MomentumStrategy()
        
        # Initialize weights
        for name in self.strategies:
            self.performance_weights[name] = 1.0
        
        logger.info(f"Registered {len(self.strategies)} strategies: {list(self.strategies.keys())}")
    
    def register_strategy(self, name: str, strategy: BaseStrategy, weight: float = 1.0):
        """
        Register a new strategy
        
        Args:
            name: Strategy name
            strategy: Strategy instance
            weight: Initial weight
        """
        self.strategies[name] = strategy
        self.performance_weights[name] = weight
        logger.info(f"Registered strategy: {name}")
    
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect current market regime
        
        Args:
            data: Market data
        
        Returns:
            Market regime: 'trending', 'ranging', 'volatile', 'quiet'
        """
        if len(data) < 50:
            return 'unknown'
        
        returns = data['close'].pct_change().dropna()
        volatility = returns.tail(50).std()
        
        # Trend detection using ADX (simplified)
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate ADX
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
        minus_di = 100 * (abs(minus_dm).ewm(alpha=1/14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/14).mean()
        
        current_adx = adx.iloc[-1] if not adx.empty else 20
        
        # Classify regime
        if current_adx > 30:
            regime = 'trending'
        elif current_adx < 20:
            regime = 'ranging'
        else:
            regime = 'transitional'
        
        # Adjust for volatility
        vol_percentile = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-8)
        if vol_percentile.iloc[-1] > 0.8:
            regime = 'volatile'
        elif vol_percentile.iloc[-1] < 0.2:
            regime = 'quiet'
        
        return regime
    
    def select_best_strategy(self, data: pd.DataFrame) -> Tuple[str, float]:
        """
        Select the best strategy for current market conditions
        
        Args:
            data: Market data
        
        Returns:
            Tuple of (strategy_name, confidence)
        """
        regime = self.detect_market_regime(data)
        
        # Strategy-regime mapping
        regime_preferences = {
            'trending': ['trend_following', 'momentum', 'mean_reversion'],
            'ranging': ['mean_reversion', 'momentum', 'trend_following'],
            'volatile': ['momentum', 'trend_following', 'mean_reversion'],
            'quiet': ['mean_reversion', 'trend_following', 'momentum'],
            'transitional': ['momentum', 'trend_following', 'mean_reversion']
        }
        
        preferences = regime_preferences.get(regime, ['trend_following', 'momentum', 'mean_reversion'])
        
        # Score each strategy
        scores = {}
        for strategy_name in preferences:
            # Base score from regime preference (position in list)
            base_score = 1.0 - (preferences.index(strategy_name) / len(preferences))
            
            # Performance weight
            perf_weight = self.performance_weights.get(strategy_name, 1.0)
            
            # Combined score
            scores[strategy_name] = base_score * perf_weight
        
        # Select best strategy
        if scores:
            best_strategy = max(scores, key=scores.get)
            confidence = scores[best_strategy]
        else:
            best_strategy = 'trend_following'
            confidence = 0.5
        
        return best_strategy, confidence
    
    def generate_signal(self, data: pd.DataFrame, 
                       position: Optional[Dict] = None) -> StrategySignal:
        """
        Generate signal using the currently selected best strategy
        
        Args:
            data: Market data
            position: Current position
        
        Returns:
            Trading signal
        """
        # Check if rotation is needed
        should_rotate = self._should_rotate()
        
        if should_rotate or self.current_strategy is None:
            # Select best strategy
            new_strategy, confidence = self.select_best_strategy(data)
            
            if new_strategy != self.current_strategy:
                self._rotate_strategy(new_strategy, confidence, data)
        
        # Generate signal from current strategy
        if self.current_strategy and self.current_strategy in self.strategies:
            strategy = self.strategies[self.current_strategy]
            signal = strategy.generate_signal(data, position)
            
            # Add rotation metadata
            signal.metadata = signal.metadata or {}
            signal.metadata['selected_strategy'] = self.current_strategy
            signal.metadata['strategy_confidence'] = self.performance_weights.get(self.current_strategy, 1.0)
            
            return signal
        else:
            # Fallback to trend following
            logger.warning("No strategy selected, using trend following")
            return self.strategies['trend_following'].generate_signal(data, position)
    
    def _should_rotate(self) -> bool:
        """Check if strategy rotation is needed"""
        if self.last_rotation is None:
            return True
        
        minutes_since_rotation = (datetime.now() - self.last_rotation).total_seconds() / 60
        return minutes_since_rotation >= self.rotation_interval
    
    def _rotate_strategy(self, new_strategy: str, confidence: float, data: pd.DataFrame):
        """
        Rotate to a new strategy
        
        Args:
            new_strategy: Name of new strategy
            confidence: Confidence in the rotation
            data: Current market data
        """
        old_strategy = self.current_strategy
        self.current_strategy = new_strategy
        self.last_rotation = datetime.now()
        
        rotation_record = {
            'timestamp': self.last_rotation,
            'from_strategy': old_strategy,
            'to_strategy': new_strategy,
            'confidence': confidence,
            'market_regime': self.detect_market_regime(data)
        }
        self.rotation_history.append(rotation_record)
        
        logger.info(f"Strategy rotated: {old_strategy} -> {new_strategy} (confidence={confidence:.2%})")
        
        # Keep only last 100 rotations
        if len(self.rotation_history) > 100:
            self.rotation_history.pop(0)
    
    def update_strategy_performance(self, strategy_name: str, 
                                    signal: StrategySignal,
                                    actual_return: float):
        """
        Update strategy performance for adaptive weighting
        
        Args:
            strategy_name: Name of strategy
            signal: The signal that was generated
            actual_return: Actual return achieved
        """
        if strategy_name in self.strategies:
            self.strategies[strategy_name].update_performance(signal, actual_return)
            
            # Update performance weight
            metrics = self.strategies[strategy_name].get_performance_metrics()
            if 'win_rate' in metrics:
                # Weight based on recent win rate and sharpe
                win_rate = metrics.get('win_rate', 0.5)
                sharpe = metrics.get('sharpe', 0)
                
                new_weight = (win_rate * 0.7 + (sharpe / 2) * 0.3)
                new_weight = max(0.5, min(2.0, new_weight * 2))
                
                # Exponential moving average of weights
                self.performance_weights[strategy_name] = (
                    0.7 * self.performance_weights.get(strategy_name, 1.0) +
                    0.3 * new_weight
                )
                
                logger.debug(f"Updated weight for {strategy_name}: {self.performance_weights[strategy_name]:.2f}")
    
    def get_performance_summary(self) -> Dict:
        """
        Get performance summary for all strategies
        
        Returns:
            Performance summary dictionary
        """
        summary = {
            'current_strategy': self.current_strategy,
            'last_rotation': self.last_rotation.isoformat() if self.last_rotation else None,
            'strategies': {}
        }
        
        for name, strategy in self.strategies.items():
            summary['strategies'][name] = {
                'weight': self.performance_weights.get(name, 1.0),
                'metrics': strategy.get_performance_metrics(),
                'signals_count': len(strategy.signals_history)
            }
        
        return summary
    
    def get_rotation_history(self, limit: int = 20) -> List[Dict]:
        """Get strategy rotation history"""
        history = []
        for record in self.rotation_history[-limit:]:
            history.append({
                'timestamp': record['timestamp'].isoformat(),
                'from_strategy': record['from_strategy'],
                'to_strategy': record['to_strategy'],
                'confidence': record['confidence'],
                'market_regime': record['market_regime']
            })
        return history
    
    def reset(self):
        """Reset all strategies and rotation history"""
        for strategy in self.strategies.values():
            strategy.reset()
        
        self.current_strategy = None
        self.last_rotation = None
        self.rotation_history = []
        
        # Reset weights
        for name in self.strategies:
            self.performance_weights[name] = 1.0
        
        logger.info("Strategy rotator reset")