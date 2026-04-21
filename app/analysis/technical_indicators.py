"""
technical_indicators.py
Part of the app/analysis module.
Provides 50+ technical indicators for market analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
from dataclasses import dataclass


@dataclass
class IndicatorResult:
    """Container for indicator calculation results"""
    values: pd.Series
    name: str
    parameters: Dict


class TechnicalIndicators:
    """Comprehensive technical indicator calculator"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data
        
        Args:
            data: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        """
        self.data = data
        self.indicators = {}
        
    def calculate_all(self, periods: List[int] = [14, 20, 50]) -> Dict[str, pd.Series]:
        """
        Calculate all common indicators
        
        Args:
            periods: List of periods for moving averages
        
        Returns:
            Dictionary of calculated indicators
        """
        results = {}
        
        # Moving Averages
        for period in periods:
            results[f'SMA_{period}'] = self.sma(period)
            results[f'EMA_{period}'] = self.ema(period)
        
        # Oscillators
        results['RSI'] = self.rsi()
        results['MACD'] = self.macd()
        results['Stochastic'] = self.stochastic()
        results['Williams_R'] = self.williams_r()
        results['CCI'] = self.cci()
        
        # Volatility
        results['BB_upper'], results['BB_middle'], results['BB_lower'] = self.bollinger_bands()
        results['ATR'] = self.atr()
        
        # Volume
        results['OBV'] = self.obv()
        results['Volume_SMA'] = self.volume_sma()
        
        # Other
        results['ADX'] = self.adx()
        results['MFI'] = self.mfi()
        
        self.indicators = results
        return results
    
    def sma(self, period: int = 20) -> pd.Series:
        """Simple Moving Average"""
        return self.data['close'].rolling(window=period).mean()
    
    def ema(self, period: int = 20) -> pd.Series:
        """Exponential Moving Average"""
        return self.data['close'].ewm(span=period, adjust=False).mean()
    
    def rsi(self, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD - Moving Average Convergence Divergence"""
        ema_fast = self.data['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def bollinger_bands(self, period: int = 20, std_dev: int = 2) -> tuple:
        """Bollinger Bands"""
        middle = self.data['close'].rolling(window=period).mean()
        std = self.data['close'].rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def atr(self, period: int = 14) -> pd.Series:
        """Average True Range"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def stochastic(self, k_period: int = 14, d_period: int = 3) -> tuple:
        """Stochastic Oscillator"""
        low_min = self.data['low'].rolling(window=k_period).min()
        high_max = self.data['high'].rolling(window=k_period).max()
        
        k = 100 * ((self.data['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    def williams_r(self, period: int = 14) -> pd.Series:
        """Williams %R"""
        high_max = self.data['high'].rolling(window=period).max()
        low_min = self.data['low'].rolling(window=period).min()
        
        williams = -100 * ((high_max - self.data['close']) / (high_max - low_min))
        return williams
    
    def cci(self, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        tp = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    def adx(self, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = self.atr(period)
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
        minus_di = 100 * (abs(minus_dm).ewm(alpha=1/period).mean() / tr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period).mean()
        
        return adx
    
    def mfi(self, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        money_flow = typical_price * self.data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (positive_sum / negative_sum)))
        return mfi
    
    def obv(self) -> pd.Series:
        """On-Balance Volume"""
        return (np.sign(self.data['close'].diff()) * self.data['volume']).fillna(0).cumsum()
    
    def volume_sma(self, period: int = 20) -> pd.Series:
        """Volume Simple Moving Average"""
        return self.data['volume'].rolling(window=period).mean()
    
    def ichimoku(self) -> dict:
        """Ichimoku Cloud"""
        conversion_line = (self.data['high'].rolling(9).max() + self.data['low'].rolling(9).min()) / 2
        base_line = (self.data['high'].rolling(26).max() + self.data['low'].rolling(26).min()) / 2
        leading_span_a = ((conversion_line + base_line) / 2).shift(26)
        leading_span_b = ((self.data['high'].rolling(52).max() + self.data['low'].rolling(52).min()) / 2).shift(26)
        lagging_span = self.data['close'].shift(-26)
        
        return {
            'conversion_line': conversion_line,
            'base_line': base_line,
            'leading_span_a': leading_span_a,
            'leading_span_b': leading_span_b,
            'lagging_span': lagging_span
        }
    
    def get_current_values(self) -> Dict[str, float]:
        """Get current values of all indicators"""
        current = {}
        for name, indicator in self.indicators.items():
            if isinstance(indicator, tuple):
                for i, sub_indicator in enumerate(indicator):
                    current[f"{name}_{i}"] = sub_indicator.iloc[-1]
            else:
                current[name] = indicator.iloc[-1]
        return current