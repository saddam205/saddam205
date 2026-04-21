"""
indicator_selector.py
AI-powered indicator selection for optimal market analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

from app.analysis.technical_indicators import TechnicalIndicators


class AutoIndicatorSelector:
    """Automatically selects best indicators based on market conditions"""
    
    def __init__(self):
        self.selected_indicators = []
        self.indicator_performance = {}
        self.market_regime = "unknown"
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_regression, k=10)
        self.performance_history = []
        
        # All available indicators
        self.all_indicators = [
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_upper', 'BB_middle', 'BB_lower',
            'ATR', 'Stochastic_K', 'Stochastic_D',
            'Williams_R', 'CCI', 'ADX', 'MFI',
            'OBV', 'Volume_SMA'
        ]
        
        # Default indicators for fallback
        self.default_indicators = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_middle']
    
    def select_best_indicators(self, data: pd.DataFrame, target_col: str = 'close') -> List[str]:
        """
        Select best indicators using machine learning
        
        Args:
            data: OHLCV data
            target_col: Target column for prediction
        
        Returns:
            List of selected indicator names
        """
        if len(data) < 100:
            return self.default_indicators
        
        try:
            # Detect market regime first
            self.market_regime = self._detect_market_regime(data)
            
            # Calculate all indicators
            indicators = TechnicalIndicators(data)
            indicator_df = self._calculate_all_indicators(indicators, data)
            
            if indicator_df.empty:
                return self.default_indicators
            
            # Prepare features and target
            features = indicator_df.dropna()
            target = data[target_col].shift(-1).dropna()
            
            # Align data
            min_len = min(len(features), len(target))
            features = features.iloc[:min_len]
            target = target.iloc[:min_len]
            
            if len(features) < 50:
                return self.default_indicators
            
            # Use Random Forest for feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(features, target)
            
            # Get feature importances
            importances = pd.Series(
                rf.feature_importances_,
                index=features.columns
            ).sort_values(ascending=False)
            
            # Store performance
            self.indicator_performance = importances.to_dict()
            
            # Select top indicators based on market regime
            selected = self._filter_by_regime(importances)
            
            # Ensure we have enough indicators
            if len(selected) < 5:
                selected = self.default_indicators
            
            self.selected_indicators = selected[:10]  # Limit to top 10
            return self.selected_indicators
            
        except Exception as e:
            print(f"Error in indicator selection: {e}")
            return self.default_indicators
    
    def _calculate_all_indicators(self, indicators: TechnicalIndicators, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all available indicators"""
        result = pd.DataFrame(index=data.index)
        
        try:
            # Moving averages
            for period in [20, 50]:
                result[f'SMA_{period}'] = indicators.sma(period)
                result[f'EMA_{period}'] = indicators.ema(period)
            
            # Oscillators
            result['RSI'] = indicators.rsi()
            macd_line, macd_signal, macd_hist = indicators.macd()
            result['MACD'] = macd_line
            result['MACD_Signal'] = macd_signal
            result['MACD_Hist'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = indicators.bollinger_bands()
            result['BB_upper'] = bb_upper
            result['BB_middle'] = bb_middle
            result['BB_lower'] = bb_lower
            
            # Other indicators
            result['ATR'] = indicators.atr()
            
            k, d = indicators.stochastic()
            result['Stochastic_K'] = k
            result['Stochastic_D'] = d
            
            result['Williams_R'] = indicators.williams_r()
            result['CCI'] = indicators.cci()
            result['ADX'] = indicators.adx()
            result['MFI'] = indicators.mfi()
            result['OBV'] = indicators.obv()
            result['Volume_SMA'] = indicators.volume_sma()
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
        
        return result
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime"""
        close = data['close']
        returns = close.pct_change().dropna()
        
        # Trend detection
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        trend_strength = abs(sma_20.iloc[-1] / sma_50.iloc[-1] - 1)
        
        # Volatility detection
        volatility = returns.std() * np.sqrt(252)
        avg_volatility = returns.rolling(252).std().mean() * np.sqrt(252)
        vol_ratio = volatility / avg_volatility if avg_volatility > 0 else 1
        
        # Range detection
        price_range = (close.max() - close.min()) / close.mean()
        
        # Classify regime
        if trend_strength > 0.05:
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                return "trending_up"
            else:
                return "trending_down"
        elif vol_ratio > 1.5:
            return "high_volatility"
        elif vol_ratio < 0.7:
            return "low_volatility"
        elif price_range < 0.05:
            return "ranging"
        else:
            return "mixed"
    
    def _filter_by_regime(self, importances: pd.Series) -> List[str]:
        """Filter indicators based on market regime"""
        regime_weights = {
            'trending_up': {
                'SMA': 1.5, 'EMA': 1.5, 'ADX': 1.3,
                'MACD': 1.2, 'RSI': 0.8, 'BB': 0.7
            },
            'trending_down': {
                'SMA': 1.5, 'EMA': 1.5, 'ADX': 1.3,
                'MACD': 1.2, 'RSI': 0.8, 'BB': 0.7
            },
            'ranging': {
                'RSI': 1.5, 'Stochastic': 1.5, 'BB': 1.3,
                'Williams': 1.3, 'CCI': 1.2, 'SMA': 0.6
            },
            'high_volatility': {
                'ATR': 1.5, 'BB': 1.3, 'ADX': 1.2,
                'Volume': 1.2, 'SMA': 0.7
            },
            'low_volatility': {
                'RSI': 1.2, 'MACD': 1.1, 'SMA': 1.0,
                'Volume': 0.8
            }
        }
        
        # Apply regime weights
        weighted_scores = importances.copy()
        regime_weight = regime_weights.get(self.market_regime, {})
        
        for indicator in weighted_scores.index:
            for key, weight in regime_weight.items():
                if key in indicator:
                    weighted_scores[indicator] *= weight
        
        # Sort and return top indicators
        return weighted_scores.sort_values(ascending=False).index.tolist()
    
    def get_indicator_performance(self) -> Dict[str, float]:
        """Get performance metrics for all indicators"""
        return self.indicator_performance
    
    def get_default_indicators(self) -> List[str]:
        """Get default indicator set"""
        return self.default_indicators
    
    def update_performance(self, indicator_returns: Dict[str, float]):
        """
        Update performance tracking for indicators
        
        Args:
            indicator_returns: Dictionary mapping indicator to its return
        """
        self.performance_history.append(indicator_returns)
        
        # Keep only last 100 records
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # Update indicator performance
        for indicator, returns in indicator_returns.items():
            if indicator in self.indicator_performance:
                # Exponential moving average of performance
                self.indicator_performance[indicator] = (
                    0.9 * self.indicator_performance[indicator] + 
                    0.1 * returns
                )
            else:
                self.indicator_performance[indicator] = returns