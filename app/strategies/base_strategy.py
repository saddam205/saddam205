"""
base_strategy.py
Part of the app/strategies module.
Base class for all trading strategies with common functionality.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    REDUCE_LONG = "REDUCE_LONG"
    REDUCE_SHORT = "REDUCE_SHORT"


@dataclass
class StrategySignal:
    """Trading signal from a strategy"""
    signal_type: SignalType
    confidence: float
    timestamp: datetime
    price: float
    reason: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'signal': self.signal_type.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'reason': self.reason,
            'metadata': self.metadata or {}
        }


@dataclass
class StrategyConfig:
    """Strategy configuration parameters"""
    name: str
    enabled: bool = True
    parameters: Dict[str, Any] = None
    weight: float = 1.0
    min_confidence: float = 0.6
    max_position_pct: float = 0.2


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Provides common functionality and interface for signal generation.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.name = config.name
        self.parameters = config.parameters or {}
        self.signals_history: List[StrategySignal] = []
        self.performance_history: List[Dict] = []
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, 
                        position: Optional[Dict] = None) -> StrategySignal:
        """
        Generate trading signal based on market data
        
        Args:
            data: OHLCV data with indicators
            position: Current position if any
        
        Returns:
            StrategySignal object
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy-specific indicators
        
        Args:
            data: Raw OHLCV data
        
        Returns:
            DataFrame with added indicators
        """
        pass
    
    def validate_signal(self, signal: StrategySignal) -> bool:
        """
        Validate signal before execution
        
        Args:
            signal: Generated signal
        
        Returns:
            Whether signal is valid
        """
        # Check confidence threshold
        if signal.confidence < self.config.min_confidence:
            logger.debug(f"Signal confidence {signal.confidence:.2%} below threshold")
            return False
        
        # Check for stale signals (older than 5 minutes)
        age = (datetime.now() - signal.timestamp).total_seconds()
        if age > 300:
            logger.debug(f"Signal is stale: {age:.0f} seconds old")
            return False
        
        return True
    
    def update_performance(self, signal: StrategySignal, 
                          actual_return: float,
                          metadata: Dict = None):
        """
        Update strategy performance tracking
        
        Args:
            signal: The signal that was generated
            actual_return: Actual return after signal
            metadata: Additional performance data
        """
        self.performance_history.append({
            'timestamp': datetime.now(),
            'signal': signal.signal_type.value,
            'confidence': signal.confidence,
            'actual_return': actual_return,
            'metadata': metadata or {}
        })
        
        # Keep only last 1000 records
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate strategy performance metrics
        
        Returns:
            Performance metrics dictionary
        """
        if not self.performance_history:
            return {'message': 'No performance data'}
        
        returns = [p['actual_return'] for p in self.performance_history]
        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r <= 0]
        
        return {
            'strategy': self.name,
            'total_signals': len(returns),
            'win_rate': len(winning) / len(returns) if returns else 0,
            'avg_return': np.mean(returns) if returns else 0,
            'avg_win': np.mean(winning) if winning else 0,
            'avg_loss': np.mean(losing) if losing else 0,
            'profit_factor': abs(sum(winning) / sum(losing)) if losing and sum(losing) != 0 else float('inf'),
            'sharpe': np.mean(returns) / np.std(returns) if returns and np.std(returns) > 0 else 0
        }
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get strategy parameter"""
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any):
        """Set strategy parameter"""
        self.parameters[key] = value
        logger.info(f"Updated parameter {key}={value} for {self.name}")
    
    def log_signal(self, signal: StrategySignal):
        """Log generated signal"""
        logger.info(f"[{self.name}] Signal: {signal.signal_type.value} "
                   f"(conf={signal.confidence:.2%}) - {signal.reason}")
        self.signals_history.append(signal)
        
        # Keep only last 500 signals
        if len(self.signals_history) > 500:
            self.signals_history.pop(0)
    
    def get_recent_signals(self, limit: int = 20) -> List[Dict]:
        """Get recent signals"""
        return [s.to_dict() for s in self.signals_history[-limit:]]
    
    def reset(self):
        """Reset strategy state"""
        self.signals_history.clear()
        self.performance_history.clear()
        logger.info(f"Strategy {self.name} reset")


class TechnicalHelper:
    """Helper class for technical calculations used by strategies"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, 
                                  std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ADX"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        atr = TechnicalHelper.calculate_atr(high, low, close, period)
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (abs(minus_dm).ewm(alpha=1/period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period).mean()
        
        return adx