"""
edge_detector.py
Part of the app/filters module.
Statistical edge detection for identifying profitable trading opportunities.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr

logger = logging.getLogger(__name__)


class EdgeType(Enum):
    """Types of statistical edges"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    SEASONAL = "seasonal"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    ARBITRAGE = "arbitrage"
    CARRY = "carry"
    BREAKOUT = "breakout"


@dataclass
class StatisticalEdge:
    """Statistical edge detection result"""
    edge_type: EdgeType
    exists: bool
    strength: float  # 0-1 scale
    confidence: float  # statistical confidence
    p_value: float
    expected_return: float
    win_rate: float
    sharpe_ratio: float
    sample_size: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'edge_type': self.edge_type.value,
            'exists': self.exists,
            'strength': self.strength,
            'confidence': self.confidence,
            'p_value': self.p_value,
            'expected_return': self.expected_return,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'sample_size': self.sample_size,
            'description': self.description
        }


class EdgeDetector:
    """
    Statistical edge detector for identifying market inefficiencies.
    Uses hypothesis testing and statistical analysis to find edges.
    """
    
    def __init__(self, min_confidence: float = 0.95, min_sample_size: int = 30):
        """
        Initialize edge detector
        
        Args:
            min_confidence: Minimum confidence level for edge detection
            min_sample_size: Minimum sample size for statistical tests
        """
        self.min_confidence = min_confidence
        self.min_sample_size = min_sample_size
        self.edge_history: List[StatisticalEdge] = []
        
    def detect_all_edges(self, data: pd.DataFrame, 
                         lookback_days: int = 252) -> List[StatisticalEdge]:
        """
        Detect all types of statistical edges
        
        Args:
            data: OHLCV data
            lookback_days: Number of days to analyze
        
        Returns:
            List of detected edges
        """
        edges = []
        
        # Detect momentum edge
        momentum_edge = self.detect_momentum_edge(data)
        if momentum_edge.exists:
            edges.append(momentum_edge)
        
        # Detect mean reversion edge
        mean_rev_edge = self.detect_mean_reversion_edge(data)
        if mean_rev_edge.exists:
            edges.append(mean_rev_edge)
        
        # Detect seasonal edge
        seasonal_edge = self.detect_seasonal_edge(data)
        if seasonal_edge.exists:
            edges.append(seasonal_edge)
        
        # Detect volatility edge
        volatility_edge = self.detect_volatility_edge(data)
        if volatility_edge.exists:
            edges.append(volatility_edge)
        
        # Detect breakout edge
        breakout_edge = self.detect_breakout_edge(data)
        if breakout_edge.exists:
            edges.append(breakout_edge)
        
        self.edge_history.extend(edges)
        
        return edges
    
    def detect_momentum_edge(self, data: pd.DataFrame, 
                            periods: List[int] = [5, 10, 20, 50]) -> StatisticalEdge:
        """
        Detect momentum edge - price continuation patterns
        
        Args:
            data: OHLCV data
            periods: Lookback periods to test
        
        Returns:
            StatisticalEdge object
        """
        returns = data['close'].pct_change().dropna()
        best_edge = None
        
        for period in periods:
            # Calculate momentum
            momentum = data['close'].pct_change(period).dropna()
            
            # Create groups: high momentum vs low momentum
            threshold = momentum.quantile(0.7)
            high_momentum = returns[momentum > threshold]
            low_momentum = returns[momentum <= threshold]
            
            if len(high_momentum) < self.min_sample_size or len(low_momentum) < self.min_sample_size:
                continue
            
            # Statistical test
            t_stat, p_value = ttest_ind(high_momentum, low_momentum, equal_var=False)
            
            if p_value < (1 - self.min_confidence):
                # Edge exists
                expected_return = high_momentum.mean() - low_momentum.mean()
                win_rate = (high_momentum > 0).mean()
                sharpe = high_momentum.mean() / high_momentum.std() * np.sqrt(252) if high_momentum.std() > 0 else 0
                
                edge = StatisticalEdge(
                    edge_type=EdgeType.MOMENTUM,
                    exists=True,
                    strength=min(1, abs(t_stat) / 10),
                    confidence=1 - p_value,
                    p_value=p_value,
                    expected_return=expected_return,
                    win_rate=win_rate,
                    sharpe_ratio=sharpe,
                    sample_size=len(high_momentum),
                    parameters={'period': period, 'threshold': threshold},
                    description=f"Momentum edge with {period}-day lookback"
                )
                
                if best_edge is None or edge.strength > best_edge.strength:
                    best_edge = edge
        
        if best_edge:
            return best_edge
        
        return StatisticalEdge(
            edge_type=EdgeType.MOMENTUM,
            exists=False,
            strength=0,
            confidence=0,
            p_value=1,
            expected_return=0,
            win_rate=0,
            sharpe_ratio=0,
            sample_size=0,
            description="No momentum edge detected"
        )
    
    def detect_mean_reversion_edge(self, data: pd.DataFrame,
                                   lookbacks: List[int] = [5, 10, 20]) -> StatisticalEdge:
        """
        Detect mean reversion edge - price reversal patterns
        
        Args:
            data: OHLCV data
            lookbacks: Lookback periods for mean calculation
        
        Returns:
            StatisticalEdge object
        """
        returns = data['close'].pct_change().dropna()
        best_edge = None
        
        for lookback in lookbacks:
            # Calculate distance from mean
            sma = data['close'].rolling(lookback).mean()
            distance = (data['close'] - sma) / sma
            future_returns = data['close'].shift(-1).pct_change()
            
            # Align data
            valid_mask = distance.notna() & future_returns.notna()
            distance_clean = distance[valid_mask]
            future_returns_clean = future_returns[valid_mask]
            
            if len(distance_clean) < self.min_sample_size:
                continue
            
            # Test correlation between distance and future returns
            corr, p_value = pearsonr(distance_clean, future_returns_clean)
            
            # Mean reversion has negative correlation
            if corr < -0.1 and p_value < (1 - self.min_confidence):
                # Create groups: oversold vs overbought
                oversold = distance_clean < distance_clean.quantile(0.2)
                overbought = distance_clean > distance_clean.quantile(0.8)
                
                oversold_returns = future_returns_clean[oversold]
                overbought_returns = future_returns_clean[overbought]
                
                expected_return = oversold_returns.mean() - overbought_returns.mean()
                win_rate = (oversold_returns > 0).mean()
                sharpe = oversold_returns.mean() / oversold_returns.std() * np.sqrt(252) if oversold_returns.std() > 0 else 0
                
                edge = StatisticalEdge(
                    edge_type=EdgeType.MEAN_REVERSION,
                    exists=True,
                    strength=min(1, abs(corr) * 5),
                    confidence=1 - p_value,
                    p_value=p_value,
                    expected_return=expected_return,
                    win_rate=win_rate,
                    sharpe_ratio=sharpe,
                    sample_size=len(oversold_returns),
                    parameters={'lookback': lookback, 'correlation': corr},
                    description=f"Mean reversion edge with {lookback}-day lookback"
                )
                
                if best_edge is None or edge.strength > best_edge.strength:
                    best_edge = edge
        
        if best_edge:
            return best_edge
        
        return StatisticalEdge(
            edge_type=EdgeType.MEAN_REVERSION,
            exists=False,
            strength=0,
            confidence=0,
            p_value=1,
            expected_return=0,
            win_rate=0,
            sharpe_ratio=0,
            sample_size=0,
            description="No mean reversion edge detected"
        )
    
    def detect_seasonal_edge(self, data: pd.DataFrame) -> StatisticalEdge:
        """
        Detect seasonal patterns - day of week, month effects
        
        Args:
            data: OHLCV data with datetime index
        
        Returns:
            StatisticalEdge object
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Data index must be datetime for seasonal detection")
            return StatisticalEdge(
                edge_type=EdgeType.SEASONAL,
                exists=False,
                strength=0,
                confidence=0,
                p_value=1,
                expected_return=0,
                win_rate=0,
                sharpe_ratio=0,
                sample_size=0,
                description="Cannot detect seasonal edge without datetime index"
            )
        
        returns = data['close'].pct_change().dropna()
        best_edge = None
        
        # Test day of week effects
        day_of_week = returns.index.dayofweek
        best_day = None
        best_return = -float('inf')
        
        for day in range(5):  # Monday to Friday
            day_returns = returns[day_of_week == day]
            other_returns = returns[day_of_week != day]
            
            if len(day_returns) < self.min_sample_size or len(other_returns) < self.min_sample_size:
                continue
            
            # Statistical test
            t_stat, p_value = ttest_ind(day_returns, other_returns, equal_var=False)
            
            if p_value < (1 - self.min_confidence) and day_returns.mean() > other_returns.mean():
                if day_returns.mean() > best_return:
                    best_return = day_returns.mean()
                    best_day = day
                    
                    edge = StatisticalEdge(
                        edge_type=EdgeType.SEASONAL,
                        exists=True,
                        strength=min(1, abs(t_stat) / 10),
                        confidence=1 - p_value,
                        p_value=p_value,
                        expected_return=day_returns.mean(),
                        win_rate=(day_returns > 0).mean(),
                        sharpe_ratio=day_returns.mean() / day_returns.std() * np.sqrt(252) if day_returns.std() > 0 else 0,
                        sample_size=len(day_returns),
                        parameters={'day_of_week': day},
                        description=f"Seasonal edge on day {day} of week"
                    )
                    best_edge = edge
        
        # Test month effects
        month = returns.index.month
        best_month = None
        best_month_return = -float('inf')
        
        for m in range(1, 13):
            month_returns = returns[month == m]
            other_returns = returns[month != m]
            
            if len(month_returns) < self.min_sample_size or len(other_returns) < self.min_sample_size:
                continue
            
            t_stat, p_value = ttest_ind(month_returns, other_returns, equal_var=False)
            
            if p_value < (1 - self.min_confidence) and month_returns.mean() > other_returns.mean():
                if month_returns.mean() > best_month_return:
                    best_month_return = month_returns.mean()
                    best_month = m
                    
                    edge = StatisticalEdge(
                        edge_type=EdgeType.SEASONAL,
                        exists=True,
                        strength=min(1, abs(t_stat) / 10),
                        confidence=1 - p_value,
                        p_value=p_value,
                        expected_return=month_returns.mean(),
                        win_rate=(month_returns > 0).mean(),
                        sharpe_ratio=month_returns.mean() / month_returns.std() * np.sqrt(252) if month_returns.std() > 0 else 0,
                        sample_size=len(month_returns),
                        parameters={'month': m},
                        description=f"Seasonal edge in month {m}"
                    )
                    
                    if best_edge is None or edge.strength > best_edge.strength:
                        best_edge = edge
        
        if best_edge:
            return best_edge
        
        return StatisticalEdge(
            edge_type=EdgeType.SEASONAL,
            exists=False,
            strength=0,
            confidence=0,
            p_value=1,
            expected_return=0,
            win_rate=0,
            sharpe_ratio=0,
            sample_size=0,
            description="No seasonal edge detected"
        )
    
    def detect_volatility_edge(self, data: pd.DataFrame) -> StatisticalEdge:
        """
        Detect volatility edge - volatility clustering patterns
        
        Args:
            data: OHLCV data
        
        Returns:
            StatisticalEdge object
        """
        returns = data['close'].pct_change().dropna()
        
        # Calculate realized volatility
        volatility = returns.rolling(20).std()
        future_returns = returns.shift(-1)
        
        # Align data
        valid_mask = volatility.notna() & future_returns.notna()
        volatility_clean = volatility[valid_mask]
        future_returns_clean = future_returns[valid_mask]
        
        if len(volatility_clean) < self.min_sample_size:
            return StatisticalEdge(
                edge_type=EdgeType.VOLATILITY,
                exists=False,
                strength=0,
                confidence=0,
                p_value=1,
                expected_return=0,
                win_rate=0,
                sharpe_ratio=0,
                sample_size=0,
                description="Insufficient data for volatility edge detection"
            )
        
        # Test correlation between volatility and future returns
        corr, p_value = pearsonr(volatility_clean, future_returns_clean)
        
        # Create groups: high volatility vs low volatility
        high_vol = volatility_clean > volatility_clean.quantile(0.7)
        low_vol = volatility_clean < volatility_clean.quantile(0.3)
        
        high_vol_returns = future_returns_clean[high_vol]
        low_vol_returns = future_returns_clean[low_vol]
        
        if len(high_vol_returns) < self.min_sample_size or len(low_vol_returns) < self.min_sample_size:
            return StatisticalEdge(
                edge_type=EdgeType.VOLATILITY,
                exists=False,
                strength=0,
                confidence=0,
                p_value=1,
                expected_return=0,
                win_rate=0,
                sharpe_ratio=0,
                sample_size=0,
                description="Insufficient samples for volatility edge"
            )
        
        # Statistical test
        t_stat, p_value = ttest_ind(high_vol_returns, low_vol_returns, equal_var=False)
        
        if p_value < (1 - self.min_confidence):
            expected_return = high_vol_returns.mean() - low_vol_returns.mean()
            
            edge = StatisticalEdge(
                edge_type=EdgeType.VOLATILITY,
                exists=True,
                strength=min(1, abs(t_stat) / 10),
                confidence=1 - p_value,
                p_value=p_value,
                expected_return=expected_return,
                win_rate=(high_vol_returns > 0).mean(),
                sharpe_ratio=high_vol_returns.mean() / high_vol_returns.std() * np.sqrt(252) if high_vol_returns.std() > 0 else 0,
                sample_size=len(high_vol_returns),
                parameters={'correlation': corr},
                description=f"Volatility edge: high volatility predicts {expected_return:+.2%} returns"
            )
            return edge
        
        return StatisticalEdge(
            edge_type=EdgeType.VOLATILITY,
            exists=False,
            strength=0,
            confidence=0,
            p_value=1,
            expected_return=0,
            win_rate=0,
            sharpe_ratio=0,
            sample_size=len(volatility_clean),
            description="No volatility edge detected"
        )
    
    def detect_breakout_edge(self, data: pd.DataFrame,
                            lookbacks: List[int] = [20, 50, 100]) -> StatisticalEdge:
        """
        Detect breakout edge - price breaking through key levels
        
        Args:
            data: OHLCV data
            lookbacks: Lookback periods for resistance/support
        
        Returns:
            StatisticalEdge object
        """
        returns = data['close'].pct_change().dropna()
        best_edge = None
        
        for lookback in lookbacks:
            # Calculate resistance and support levels
            rolling_high = data['high'].rolling(lookback).max()
            rolling_low = data['low'].rolling(lookback).min()
            
            # Detect breakouts
            resistance_break = data['close'] > rolling_high.shift(1)
            support_break = data['close'] < rolling_low.shift(1)
            
            # Calculate future returns after breakout
            future_returns = data['close'].shift(-5).pct_change(5)  # 5-day forward return
            
            # Resistance breakouts
            valid_mask = resistance_break.notna() & future_returns.notna()
            breakout_returns = future_returns[valid_mask][resistance_break[valid_mask]]
            non_breakout_returns = future_returns[valid_mask][~resistance_break[valid_mask]]
            
            if len(breakout_returns) >= self.min_sample_size and len(non_breakout_returns) >= self.min_sample_size:
                t_stat, p_value = ttest_ind(breakout_returns, non_breakout_returns, equal_var=False)
                
                if p_value < (1 - self.min_confidence) and breakout_returns.mean() > 0:
                    edge = StatisticalEdge(
                        edge_type=EdgeType.BREAKOUT,
                        exists=True,
                        strength=min(1, abs(t_stat) / 10),
                        confidence=1 - p_value,
                        p_value=p_value,
                        expected_return=breakout_returns.mean(),
                        win_rate=(breakout_returns > 0).mean(),
                        sharpe_ratio=breakout_returns.mean() / breakout_returns.std() * np.sqrt(252) if breakout_returns.std() > 0 else 0,
                        sample_size=len(breakout_returns),
                        parameters={'lookback': lookback, 'type': 'resistance'},
                        description=f"Resistance breakout edge with {lookback}-day lookback"
                    )
                    
                    if best_edge is None or edge.strength > best_edge.strength:
                        best_edge = edge
            
            # Support breakouts (downward)
            valid_mask = support_break.notna() & future_returns.notna()
            breakout_returns_down = future_returns[valid_mask][support_break[valid_mask]]
            
            if len(breakout_returns_down) >= self.min_sample_size:
                t_stat, p_value = ttest_ind(breakout_returns_down, non_breakout_returns, equal_var=False)
                
                if p_value < (1 - self.min_confidence) and breakout_returns_down.mean() < 0:
                    edge = StatisticalEdge(
                        edge_type=EdgeType.BREAKOUT,
                        exists=True,
                        strength=min(1, abs(t_stat) / 10),
                        confidence=1 - p_value,
                        p_value=p_value,
                        expected_return=breakout_returns_down.mean(),
                        win_rate=(breakout_returns_down > 0).mean(),
                        sharpe_ratio=breakout_returns_down.mean() / breakout_returns_down.std() * np.sqrt(252) if breakout_returns_down.std() > 0 else 0,
                        sample_size=len(breakout_returns_down),
                        parameters={'lookback': lookback, 'type': 'support'},
                        description=f"Support breakdown edge with {lookback}-day lookback"
                    )
                    
                    if best_edge is None or edge.strength > best_edge.strength:
                        best_edge = edge
        
        if best_edge:
            return best_edge
        
        return StatisticalEdge(
            edge_type=EdgeType.BREAKOUT,
            exists=False,
            strength=0,
            confidence=0,
            p_value=1,
            expected_return=0,
            win_rate=0,
            sharpe_ratio=0,
            sample_size=0,
            description="No breakout edge detected"
        )
    
    def get_best_edge(self) -> Optional[StatisticalEdge]:
        """Get the strongest detected edge"""
        if not self.edge_history:
            return None
        
        return max(self.edge_history, key=lambda e: e.strength if e.exists else 0)
    
    def get_edge_summary(self) -> Dict:
        """Get summary of all detected edges"""
        return {
            'total_edges': len(self.edge_history),
            'edges_by_type': {
                edge.edge_type.value: edge.to_dict()
                for edge in self.edge_history
                if edge.exists
            },
            'best_edge': self.get_best_edge().to_dict() if self.get_best_edge() else None,
            'overall_edge_score': np.mean([e.strength for e in self.edge_history if e.exists]) if self.edge_history else 0
        }