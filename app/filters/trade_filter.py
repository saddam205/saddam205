"""
trade_filter.py
Part of the app/filters module.
Multi-layer trade filtering system for signal validation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FilterCondition(Enum):
    """Types of filter conditions"""
    MIN_CONFIDENCE = "min_confidence"
    MAX_VOLATILITY = "max_volatility"
    MIN_VOLUME = "min_volume"
    MAX_SPREAD = "max_spread"
    TREND_ALIGNMENT = "trend_alignment"
    REGIME_FILTER = "regime_filter"
    TIME_FILTER = "time_filter"
    CORRELATION_FILTER = "correlation_filter"
    MAX_POSITION = "max_position"
    DAILY_LIMIT = "daily_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    CUSTOM = "custom"


@dataclass
class FilterResult:
    """Result of applying a filter"""
    passed: bool
    condition: FilterCondition
    reason: str
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'passed': self.passed,
            'condition': self.condition.value,
            'reason': self.reason,
            'score': self.score,
            'metadata': self.metadata
        }


@dataclass
class TradeSignal:
    """Trade signal to be filtered"""
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    timestamp: datetime
    predicted_return: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradeFilter:
    """
    Multi-layer trade filtering system.
    Applies multiple filters to validate trading signals before execution.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trade filter
        
        Args:
            config: Filter configuration
        """
        self.config = config or self._default_config()
        self.filters: List[Tuple[FilterCondition, Callable]] = []
        self.filter_history: List[Dict] = []
        self.statistics = {
            'total_signals': 0,
            'passed_signals': 0,
            'rejected_signals': 0,
            'rejection_reasons': {}
        }
        
        # Register default filters
        self._register_default_filters()
        
    def _default_config(self) -> Dict:
        """Default filter configuration"""
        return {
            'min_confidence': 0.65,  # Minimum 65% confidence
            'max_volatility': 0.05,  # Maximum 5% daily volatility
            'min_volume_ratio': 1.5,  # Minimum 1.5x average volume
            'max_spread_bps': 10,  # Maximum 10 bps spread
            'require_trend_alignment': True,
            'allowed_regimes': ['trending_up', 'trending_down', 'breakout'],
            'forbidden_hours': [],  # 24/7 trading by default
            'max_daily_trades': 10,
            'max_daily_loss_percent': 5,
            'max_drawdown_percent': 15,
            'require_liquidity': True,
            'min_liquidity_usd': 1000000  # $1M minimum liquidity
        }
    
    def _register_default_filters(self):
        """Register default filter functions"""
        self.register_filter(FilterCondition.MIN_CONFIDENCE, self._confidence_filter)
        self.register_filter(FilterCondition.MAX_VOLATILITY, self._volatility_filter)
        self.register_filter(FilterCondition.MIN_VOLUME, self._volume_filter)
        self.register_filter(FilterCondition.MAX_SPREAD, self._spread_filter)
        self.register_filter(FilterCondition.TREND_ALIGNMENT, self._trend_filter)
        self.register_filter(FilterCondition.REGIME_FILTER, self._regime_filter)
        self.register_filter(FilterCondition.TIME_FILTER, self._time_filter)
        self.register_filter(FilterCondition.MAX_POSITION, self._position_filter)
        self.register_filter(FilterCondition.DAILY_LIMIT, self._daily_limit_filter)
        self.register_filter(FilterCondition.DRAWDOWN_LIMIT, self._drawdown_filter)
    
    def register_filter(self, condition: FilterCondition, 
                        filter_func: Callable) -> None:
        """
        Register a custom filter
        
        Args:
            condition: Filter condition type
            filter_func: Filter function that returns FilterResult
        """
        self.filters.append((condition, filter_func))
        logger.info(f"Registered filter: {condition.value}")
    
    def filter_signal(self, signal: TradeSignal, market_data: pd.DataFrame,
                     portfolio_state: Dict) -> Tuple[bool, List[FilterResult]]:
        """
        Apply all filters to a trading signal
        
        Args:
            signal: Trade signal to filter
            market_data: Current market data
            portfolio_state: Current portfolio state
        
        Returns:
            Tuple of (passed, list_of_filter_results)
        """
        self.statistics['total_signals'] += 1
        results = []
        
        for condition, filter_func in self.filters:
            try:
                result = filter_func(signal, market_data, portfolio_state)
                results.append(result)
                
                if not result.passed:
                    self._record_rejection(condition, result.reason)
                    logger.debug(f"Signal rejected by {condition.value}: {result.reason}")
                    return False, results
                    
            except Exception as e:
                logger.error(f"Filter {condition.value} failed: {e}")
                result = FilterResult(
                    passed=False,
                    condition=condition,
                    reason=f"Filter error: {str(e)}",
                    score=0
                )
                results.append(result)
                return False, results
        
        # All filters passed
        self.statistics['passed_signals'] += 1
        logger.info(f"Signal passed all filters: {signal.symbol} {signal.signal}")
        return True, results
    
    def _confidence_filter(self, signal: TradeSignal, market_data: pd.DataFrame,
                          portfolio_state: Dict) -> FilterResult:
        """Filter based on signal confidence"""
        min_conf = self.config.get('min_confidence', 0.65)
        
        if signal.confidence >= min_conf:
            return FilterResult(
                passed=True,
                condition=FilterCondition.MIN_CONFIDENCE,
                reason=f"Confidence {signal.confidence:.1%} >= {min_conf:.1%}",
                score=signal.confidence
            )
        else:
            return FilterResult(
                passed=False,
                condition=FilterCondition.MIN_CONFIDENCE,
                reason=f"Confidence {signal.confidence:.1%} < {min_conf:.1%}",
                score=signal.confidence
            )
    
    def _volatility_filter(self, signal: TradeSignal, market_data: pd.DataFrame,
                          portfolio_state: Dict) -> FilterResult:
        """Filter based on market volatility"""
        max_vol = self.config.get('max_volatility', 0.05)
        
        # Calculate recent volatility
        if len(market_data) >= 20:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.tail(20).std() * np.sqrt(252)
        else:
            volatility = 0.03  # Default
        
        if volatility <= max_vol:
            return FilterResult(
                passed=True,
                condition=FilterCondition.MAX_VOLATILITY,
                reason=f"Volatility {volatility:.2%} <= {max_vol:.2%}",
                score=1 - (volatility / max_vol)
            )
        else:
            return FilterResult(
                passed=False,
                condition=FilterCondition.MAX_VOLATILITY,
                reason=f"Volatility {volatility:.2%} > {max_vol:.2%}",
                score=0
            )
    
    def _volume_filter(self, signal: TradeSignal, market_data: pd.DataFrame,
                      portfolio_state: Dict) -> FilterResult:
        """Filter based on trading volume"""
        min_volume_ratio = self.config.get('min_volume_ratio', 1.5)
        
        if len(market_data) >= 20:
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].tail(20).mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio >= min_volume_ratio:
                return FilterResult(
                    passed=True,
                    condition=FilterCondition.MIN_VOLUME,
                    reason=f"Volume ratio {volume_ratio:.1f}x >= {min_volume_ratio:.1f}x",
                    score=min(1, volume_ratio / min_volume_ratio)
                )
            else:
                return FilterResult(
                    passed=False,
                    condition=FilterCondition.MIN_VOLUME,
                    reason=f"Volume ratio {volume_ratio:.1f}x < {min_volume_ratio:.1f}x",
                    score=0
                )
        
        return FilterResult(
            passed=True,
            condition=FilterCondition.MIN_VOLUME,
            reason="Insufficient data for volume filter",
            score=0.5
        )
    
    def _spread_filter(self, signal: TradeSignal, market_data: pd.DataFrame,
                      portfolio_state: Dict) -> FilterResult:
        """Filter based on bid-ask spread"""
        max_spread_bps = self.config.get('max_spread_bps', 10)
        
        # Calculate spread from OHLC data (approximation)
        if 'high' in market_data.columns and 'low' in market_data.columns:
            current_high = market_data['high'].iloc[-1]
            current_low = market_data['low'].iloc[-1]
            spread_bps = ((current_high - current_low) / signal.price) * 10000
        else:
            spread_bps = 5  # Default assumption
        
        if spread_bps <= max_spread_bps:
            return FilterResult(
                passed=True,
                condition=FilterCondition.MAX_SPREAD,
                reason=f"Spread {spread_bps:.1f}bps <= {max_spread_bps}bps",
                score=1 - (spread_bps / max_spread_bps)
            )
        else:
            return FilterResult(
                passed=False,
                condition=FilterCondition.MAX_SPREAD,
                reason=f"Spread {spread_bps:.1f}bps > {max_spread_bps}bps",
                score=0
            )
    
    def _trend_filter(self, signal: TradeSignal, market_data: pd.DataFrame,
                     portfolio_state: Dict) -> FilterResult:
        """Filter based on trend alignment"""
        if not self.config.get('require_trend_alignment', True):
            return FilterResult(
                passed=True,
                condition=FilterCondition.TREND_ALIGNMENT,
                reason="Trend alignment not required",
                score=1
            )
        
        if len(market_data) < 50:
            return FilterResult(
                passed=True,
                condition=FilterCondition.TREND_ALIGNMENT,
                reason="Insufficient data",
                score=0.5
            )
        
        # Calculate trend
        sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
        sma_50 = market_data['close'].rolling(50).mean().iloc[-1]
        
        trend = "UP" if sma_20 > sma_50 else "DOWN"
        
        # Check alignment with signal
        if signal.signal == "BUY" and trend == "UP":
            return FilterResult(
                passed=True,
                condition=FilterCondition.TREND_ALIGNMENT,
                reason=f"Buy signal aligns with {trend} trend",
                score=1
            )
        elif signal.signal == "SELL" and trend == "DOWN":
            return FilterResult(
                passed=True,
                condition=FilterCondition.TREND_ALIGNMENT,
                reason=f"Sell signal aligns with {trend} trend",
                score=1
            )
        elif signal.signal == "HOLD":
            return FilterResult(
                passed=True,
                condition=FilterCondition.TREND_ALIGNMENT,
                reason="Hold signal always passes",
                score=0.5
            )
        else:
            return FilterResult(
                passed=False,
                condition=FilterCondition.TREND_ALIGNMENT,
                reason=f"Signal {signal.signal} against {trend} trend",
                score=0
            )
    
    def _regime_filter(self, signal: TradeSignal, market_data: pd.DataFrame,
                      portfolio_state: Dict) -> FilterResult:
        """Filter based on market regime"""
        allowed_regimes = self.config.get('allowed_regimes', [])
        
        if not allowed_regimes:
            return FilterResult(
                passed=True,
                condition=FilterCondition.REGIME_FILTER,
                reason="No regime restrictions",
                score=1
            )
        
        # Detect market regime (simplified)
        if len(market_data) >= 50:
            sma_20 = market_data['close'].rolling(20).mean()
            sma_50 = market_data['close'].rolling(50).mean()
            
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                regime = "trending_up"
            else:
                regime = "trending_down"
            
            # Check volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std()
            if volatility > 0.03:
                regime = "high_volatility"
            elif volatility < 0.01:
                regime = "low_volatility"
        else:
            regime = "unknown"
        
        if regime in allowed_regimes:
            return FilterResult(
                passed=True,
                condition=FilterCondition.REGIME_FILTER,
                reason=f"Regime {regime} is allowed",
                score=1
            )
        else:
            return FilterResult(
                passed=False,
                condition=FilterCondition.REGIME_FILTER,
                reason=f"Regime {regime} not in allowed list",
                score=0
            )
    
    def _time_filter(self, signal: TradeSignal, market_data: pd.DataFrame,
                    portfolio_state: Dict) -> FilterResult:
        """Filter based on time of day"""
        forbidden_hours = self.config.get('forbidden_hours', [])
        
        if not forbidden_hours:
            return FilterResult(
                passed=True,
                condition=FilterCondition.TIME_FILTER,
                reason="No time restrictions",
                score=1
            )
        
        current_hour = signal.timestamp.hour
        
        if current_hour not in forbidden_hours:
            return FilterResult(
                passed=True,
                condition=FilterCondition.TIME_FILTER,
                reason=f"Hour {current_hour} is allowed",
                score=1
            )
        else:
            return FilterResult(
                passed=False,
                condition=FilterCondition.TIME_FILTER,
                reason=f"Hour {current_hour} is forbidden",
                score=0
            )
    
    def _position_filter(self, signal: TradeSignal, market_data: pd.DataFrame,
                        portfolio_state: Dict) -> FilterResult:
        """Filter based on current positions"""
        max_positions = self.config.get('max_positions', 5)
        current_positions = portfolio_state.get('open_positions', 0)
        
        if current_positions < max_positions:
            return FilterResult(
                passed=True,
                condition=FilterCondition.MAX_POSITION,
                reason=f"Position count {current_positions} < {max_positions}",
                score=1 - (current_positions / max_positions)
            )
        else:
            return FilterResult(
                passed=False,
                condition=FilterCondition.MAX_POSITION,
                reason=f"Maximum positions ({max_positions}) reached",
                score=0
            )
    
    def _daily_limit_filter(self, signal: TradeSignal, market_data: pd.DataFrame,
                           portfolio_state: Dict) -> FilterResult:
        """Filter based on daily trading limits"""
        max_daily_trades = self.config.get('max_daily_trades', 10)
        today_trades = portfolio_state.get('today_trades', 0)
        
        if today_trades < max_daily_trades:
            return FilterResult(
                passed=True,
                condition=FilterCondition.DAILY_LIMIT,
                reason=f"Daily trades {today_trades} < {max_daily_trades}",
                score=1 - (today_trades / max_daily_trades)
            )
        else:
            return FilterResult(
                passed=False,
                condition=FilterCondition.DAILY_LIMIT,
                reason=f"Daily limit ({max_daily_trades}) reached",
                score=0
            )
    
    def _drawdown_filter(self, signal: TradeSignal, market_data: pd.DataFrame,
                        portfolio_state: Dict) -> FilterResult:
        """Filter based on drawdown limits"""
        max_drawdown = self.config.get('max_drawdown_percent', 15)
        current_drawdown = portfolio_state.get('current_drawdown', 0)
        
        if current_drawdown <= max_drawdown:
            return FilterResult(
                passed=True,
                condition=FilterCondition.DRAWDOWN_LIMIT,
                reason=f"Drawdown {current_drawdown:.1f}% <= {max_drawdown}%",
                score=1 - (current_drawdown / max_drawdown)
            )
        else:
            return FilterResult(
                passed=False,
                condition=FilterCondition.DRAWDOWN_LIMIT,
                reason=f"Drawdown {current_drawdown:.1f}% > {max_drawdown}%",
                score=0
            )
    
    def _record_rejection(self, condition: FilterCondition, reason: str):
        """Record rejection for statistics"""
        self.statistics['rejected_signals'] += 1
        condition_key = condition.value
        self.statistics['rejection_reasons'][condition_key] = \
            self.statistics['rejection_reasons'].get(condition_key, 0) + 1
    
    def get_statistics(self) -> Dict:
        """Get filter statistics"""
        total = self.statistics['total_signals']
        passed = self.statistics['passed_signals']
        
        return {
            **self.statistics,
            'pass_rate': passed / total if total > 0 else 0,
            'reject_rate': self.statistics['rejected_signals'] / total if total > 0 else 0,
            'top_rejection_reasons': dict(
                sorted(self.statistics['rejection_reasons'].items(), 
                      key=lambda x: x[1], reverse=True)[:5]
            )
        }
    
    def update_config(self, new_config: Dict) -> None:
        """Update filter configuration"""
        self.config.update(new_config)
        logger.info(f"Filter config updated: {new_config}")
    
    def reset_statistics(self) -> None:
        """Reset filter statistics"""
        self.statistics = {
            'total_signals': 0,
            'passed_signals': 0,
            'rejected_signals': 0,
            'rejection_reasons': {}
        }
        logger.info("Filter statistics reset")