"""
__init__.py
Part of the app/strategies module.
Exports trading strategies for market analysis and signal generation.
"""

from .base_strategy import BaseStrategy, StrategySignal, SignalType
from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .strategy_rotator import StrategyRotator, RotationCriteria

__all__ = [
    'BaseStrategy',
    'StrategySignal',
    'SignalType',
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'StrategyRotator',
    'RotationCriteria'
]