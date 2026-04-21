"""
trend_following.py
Part of the app/strategies module.
Trend following strategy using moving averages and ADX.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

from .base_strategy import BaseStrategy, StrategyConfig, StrategySignal, SignalType, TechnicalHelper

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy using:
    - Dual moving average crossover
    - ADX for trend strength
    - Price position relative to moving averages
    """
    
    def __init__(self, config: StrategyConfig = None):
        """
        Initialize trend following strategy
        
        Args:
            config: Strategy configuration
        """
        if config is None:
            config = StrategyConfig(
                name="TrendFollowing",
                parameters={
                    'fast_ma': 20,
                    'slow_ma': 50,
                    'adx_threshold': 25,
                    'trend_strength_min': 0.02,
                    'use_volume_confirmation': True
                },
                min_confidence=0.6,
                weight=1.0
            )
        super().__init__(config)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend following indicators
        
        Args:
            data: OHLCV data
        
        Returns:
            DataFrame with added indicators
        """
        df = data.copy()
        
        # Moving averages
        fast_ma = self.get_parameter('fast_ma', 20)
        slow_ma = self.get_parameter('slow_ma', 50)
        
        df['fast_ma'] = TechnicalHelper.calculate_sma(df['close'], fast_ma)
        df['slow_ma'] = TechnicalHelper.calculate_sma(df['close'], slow_ma)
        
        # ADX for trend strength
        df['adx'] = TechnicalHelper.calculate_adx(
            df['high'], df['low'], df['close'], period=14
        )
        
        # Price position relative to MAs
        df['price_vs_fast'] = (df['close'] - df['fast_ma']) / df['fast_ma']
        df['price_vs_slow'] = (df['close'] - df['slow_ma']) / df['slow_ma']
        
        # MA slope
        df['fast_slope'] = df['fast_ma'].diff(5) / df['fast_ma']
        df['slow_slope'] = df['slow_ma'].diff(5) / df['slow_ma']
        
        # Volume confirmation
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def generate_signal(self, data: pd.DataFrame, 
                        position: Optional[Dict] = None) -> StrategySignal:
        """
        Generate trend following signal
        
        Args:
            data: OHLCV data with indicators
            position: Current position (optional)
        
        Returns:
            Trading signal
        """
        if len(data) < self.get_parameter('slow_ma', 50):
            return StrategySignal(
                signal_type=SignalType.HOLD,
                confidence=0.5,
                timestamp=datetime.now(),
                price=data['close'].iloc[-1],
                reason="Insufficient data"
            )
        
        df = self.calculate_indicators(data)
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        
        current_price = last['close']
        fast_ma = last['fast_ma']
        slow_ma = last['slow_ma']
        adx = last['adx']
        adx_threshold = self.get_parameter('adx_threshold', 25)
        
        # Check for crossover signals
        fast_above_slow = fast_ma > slow_ma
        prev_fast_above_slow = prev['fast_ma'] > prev['slow_ma']
        
        # Golden cross (fast crosses above slow)
        if not prev_fast_above_slow and fast_above_slow:
            if adx > adx_threshold:
                confidence = min(0.85, 0.6 + (adx - adx_threshold) / 100)
                return StrategySignal(
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    price=current_price,
                    reason=f"Golden cross: Fast MA({fast_ma:.2f}) > Slow MA({slow_ma:.2f}), ADX={adx:.1f}"
                )
        
        # Death cross (fast crosses below slow)
        elif prev_fast_above_slow and not fast_above_slow:
            if adx > adx_threshold:
                confidence = min(0.85, 0.6 + (adx - adx_threshold) / 100)
                return StrategySignal(
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    price=current_price,
                    reason=f"Death cross: Fast MA({fast_ma:.2f}) < Slow MA({slow_ma:.2f}), ADX={adx:.1f}"
                )
        
        # Trend continuation signals
        if fast_above_slow and adx > adx_threshold:
            # Uptrend continuation
            price_vs_fast = last['price_vs_fast']
            if price_vs_fast > -0.01 and last['fast_slope'] > 0:
                confidence = 0.6 + min(0.2, adx / 100)
                return StrategySignal(
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    price=current_price,
                    reason=f"Uptrend continuation: ADX={adx:.1f}, slope positive"
                )
        
        elif not fast_above_slow and adx > adx_threshold:
            # Downtrend continuation
            price_vs_fast = last['price_vs_fast']
            if price_vs_fast < 0.01 and last['fast_slope'] < 0:
                confidence = 0.6 + min(0.2, adx / 100)
                return StrategySignal(
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    price=current_price,
                    reason=f"Downtrend continuation: ADX={adx:.1f}, slope negative"
                )
        
        # No clear signal
        return StrategySignal(
            signal_type=SignalType.HOLD,
            confidence=0.5,
            timestamp=datetime.now(),
            price=current_price,
            reason=f"No clear trend: ADX={adx:.1f}, MA spread={(fast_ma/slow_ma-1):.2%}"
        )