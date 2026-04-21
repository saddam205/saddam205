"""
momentum.py
Part of the app/strategies module.
Momentum strategy using rate of change and MACD.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

from .base_strategy import BaseStrategy, StrategyConfig, StrategySignal, SignalType, TechnicalHelper

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using:
    - Rate of Change (ROC)
    - MACD for momentum direction
    - Volume confirmation
    - Relative strength ranking
    """
    
    def __init__(self, config: StrategyConfig = None):
        """
        Initialize momentum strategy
        
        Args:
            config: Strategy configuration
        """
        if config is None:
            config = StrategyConfig(
                name="Momentum",
                parameters={
                    'roc_period': 14,
                    'roc_threshold': 5,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'momentum_lookback': 20,
                    'volume_confirm': True,
                    'min_momentum_strength': 0.3
                },
                min_confidence=0.6,
                weight=1.0
            )
        super().__init__(config)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators
        
        Args:
            data: OHLCV data
        
        Returns:
            DataFrame with added indicators
        """
        df = data.copy()
        
        # Rate of Change
        roc_period = self.get_parameter('roc_period', 14)
        df['roc'] = (df['close'] / df['close'].shift(roc_period) - 1) * 100
        
        # MACD
        fast = self.get_parameter('macd_fast', 12)
        slow = self.get_parameter('macd_slow', 26)
        signal = self.get_parameter('macd_signal', 9)
        df['macd'], df['macd_signal'], df['macd_hist'] = TechnicalHelper.calculate_macd(
            df['close'], fast, slow, signal
        )
        
        # Momentum score
        df['momentum_score'] = self._calculate_momentum_score(df)
        
        # Volume momentum
        df['volume_momentum'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Price momentum
        lookback = self.get_parameter('momentum_lookback', 20)
        df['price_momentum'] = (df['close'] - df['close'].shift(lookback)) / df['close'].shift(lookback) * 100
        
        return df
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite momentum score"""
        # Normalize ROC to 0-1 range
        roc_norm = (df['roc'] - df['roc'].min()) / (df['roc'].max() - df['roc'].min() + 1e-8)
        
        # MACD histogram strength
        macd_norm = (df['macd_hist'] - df['macd_hist'].min()) / (df['macd_hist'].max() - df['macd_hist'].min() + 1e-8)
        
        # Combine scores
        score = (roc_norm * 0.5 + macd_norm * 0.5)
        return score
    
    def generate_signal(self, data: pd.DataFrame,
                        position: Optional[Dict] = None) -> StrategySignal:
        """
        Generate momentum signal
        
        Args:
            data: OHLCV data with indicators
            position: Current position (optional)
        
        Returns:
            Trading signal
        """
        if len(data) < self.get_parameter('momentum_lookback', 20):
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
        roc = last['roc']
        roc_threshold = self.get_parameter('roc_threshold', 5)
        min_strength = self.get_parameter('min_momentum_strength', 0.3)
        
        # MACD crossover detection
        macd_above_signal = last['macd'] > last['macd_signal']
        prev_macd_above = prev['macd'] > prev['macd_signal']
        
        # Bullish momentum
        if (macd_above_signal and not prev_macd_above and roc > 0):
            confidence = 0.6 + min(0.3, abs(last['macd_hist']) / 100)
            if last['momentum_score'] > min_strength:
                return StrategySignal(
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    price=current_price,
                    reason=f"Bullish MACD crossover, ROC={roc:.1f}%, Momentum score={last['momentum_score']:.2f}"
                )
        
        # Bearish momentum
        elif (not macd_above_signal and prev_macd_above and roc < 0):
            confidence = 0.6 + min(0.3, abs(last['macd_hist']) / 100)
            if last['momentum_score'] > min_strength:
                return StrategySignal(
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    price=current_price,
                    reason=f"Bearish MACD crossover, ROC={roc:.1f}%, Momentum score={last['momentum_score']:.2f}"
                )
        
        # Strong momentum continuation
        volume_confirm = self.get_parameter('volume_confirm', True)
        if volume_confirm and last['volume_momentum'] > 1.2:
            if roc > roc_threshold:
                confidence = 0.6 + min(0.3, roc / 50)
                return StrategySignal(
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    price=current_price,
                    reason=f"Strong upside momentum: ROC={roc:.1f}%, Volume surge"
                )
            elif roc < -roc_threshold:
                confidence = 0.6 + min(0.3, abs(roc) / 50)
                return StrategySignal(
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    price=current_price,
                    reason=f"Strong downside momentum: ROC={roc:.1f}%, Volume surge"
                )
        
        return StrategySignal(
            signal_type=SignalType.HOLD,
            confidence=0.5,
            timestamp=datetime.now(),
            price=current_price,
            reason=f"Momentum neutral: ROC={roc:.1f}%, MACD={last['macd']:.2f}"
        )