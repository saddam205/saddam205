"""
mean_reversion.py
Part of the app/strategies module.
Mean reversion strategy using RSI and Bollinger Bands.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

from .base_strategy import BaseStrategy, StrategyConfig, StrategySignal, SignalType, TechnicalHelper

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using:
    - RSI for overbought/oversold conditions
    - Bollinger Bands for price extremes
    - Z-score for deviation measurement
    """
    
    def __init__(self, config: StrategyConfig = None):
        """
        Initialize mean reversion strategy
        
        Args:
            config: Strategy configuration
        """
        if config is None:
            config = StrategyConfig(
                name="MeanReversion",
                parameters={
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70,
                    'bb_period': 20,
                    'bb_std': 2,
                    'z_score_threshold': 2.0,
                    'mean_reversion_strength': 0.7
                },
                min_confidence=0.55,
                weight=1.0
            )
        super().__init__(config)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean reversion indicators
        
        Args:
            data: OHLCV data
        
        Returns:
            DataFrame with added indicators
        """
        df = data.copy()
        
        # RSI
        rsi_period = self.get_parameter('rsi_period', 14)
        df['rsi'] = TechnicalHelper.calculate_rsi(df['close'], rsi_period)
        
        # Bollinger Bands
        bb_period = self.get_parameter('bb_period', 20)
        bb_std = self.get_parameter('bb_std', 2)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = TechnicalHelper.calculate_bollinger_bands(
            df['close'], bb_period, bb_std
        )
        
        # Z-score (deviation from mean)
        rolling_mean = df['close'].rolling(bb_period).mean()
        rolling_std = df['close'].rolling(bb_period).std()
        df['z_score'] = (df['close'] - rolling_mean) / rolling_std
        
        # Distance from bands
        df['dist_to_upper'] = (df['close'] - df['bb_upper']) / df['bb_upper']
        df['dist_to_lower'] = (df['bb_lower'] - df['close']) / df['bb_lower']
        
        # Mean reversion strength
        df['reversion_score'] = self._calculate_reversion_score(df)
        
        return df
    
    def _calculate_reversion_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate mean reversion strength score"""
        # Higher score = stronger mean reversion expected
        rsi_normalized = (50 - df['rsi'].abs()) / 50
        z_score_norm = 1 - min(1, abs(df['z_score']) / 3)
        
        score = (rsi_normalized * 0.5 + z_score_norm * 0.5)
        return score
    
    def generate_signal(self, data: pd.DataFrame,
                        position: Optional[Dict] = None) -> StrategySignal:
        """
        Generate mean reversion signal
        
        Args:
            data: OHLCV data with indicators
            position: Current position (optional)
        
        Returns:
            Trading signal
        """
        if len(data) < self.get_parameter('bb_period', 20):
            return StrategySignal(
                signal_type=SignalType.HOLD,
                confidence=0.5,
                timestamp=datetime.now(),
                price=data['close'].iloc[-1],
                reason="Insufficient data"
            )
        
        df = self.calculate_indicators(data)
        last = df.iloc[-1]
        
        current_price = last['close']
        rsi = last['rsi']
        rsi_oversold = self.get_parameter('rsi_oversold', 30)
        rsi_overbought = self.get_parameter('rsi_overbought', 70)
        z_score_threshold = self.get_parameter('z_score_threshold', 2.0)
        reversion_strength = self.get_parameter('mean_reversion_strength', 0.7)
        
        # Oversold condition (potential buy)
        if rsi < rsi_oversold or last['z_score'] < -z_score_threshold:
            confidence = 0.5 + (rsi_oversold - rsi) / 100
            confidence = min(0.85, confidence * reversion_strength)
            
            if confidence >= self.config.min_confidence:
                return StrategySignal(
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    price=current_price,
                    reason=f"Oversold: RSI={rsi:.1f}, Z-score={last['z_score']:.2f}"
                )
        
        # Overbought condition (potential sell)
        elif rsi > rsi_overbought or last['z_score'] > z_score_threshold:
            confidence = 0.5 + (rsi - rsi_overbought) / 100
            confidence = min(0.85, confidence * reversion_strength)
            
            if confidence >= self.config.min_confidence:
                return StrategySignal(
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    price=current_price,
                    reason=f"Overbought: RSI={rsi:.1f}, Z-score={last['z_score']:.2f}"
                )
        
        # Exit signals for existing positions
        if position:
            if position.get('side') == 'LONG' and rsi > 50:
                return StrategySignal(
                    signal_type=SignalType.CLOSE_LONG,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    price=current_price,
                    reason=f"Exiting long: RSI reverted to {rsi:.1f}"
                )
            elif position.get('side') == 'SHORT' and rsi < 50:
                return StrategySignal(
                    signal_type=SignalType.CLOSE_SHORT,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    price=current_price,
                    reason=f"Exiting short: RSI reverted to {rsi:.1f}"
                )
        
        return StrategySignal(
            signal_type=SignalType.HOLD,
            confidence=0.5,
            timestamp=datetime.now(),
            price=current_price,
            reason=f"Neutral: RSI={rsi:.1f}, Z-score={last['z_score']:.2f}"
        )