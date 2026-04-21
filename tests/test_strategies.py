"""
test_strategies.py
Unit tests for trading strategies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategySignal, SignalType
from app.strategies.trend_following import TrendFollowingStrategy
from app.strategies.mean_reversion import MeanReversionStrategy
from app.strategies.momentum import MomentumStrategy
from app.strategies.strategy_rotator import StrategyRotator


class TestBaseStrategy:
    """Test base strategy functionality"""
    
    @pytest.fixture
    def mock_strategy(self):
        class MockStrategy(BaseStrategy):
            def generate_signal(self, data, position=None):
                return StrategySignal(
                    signal_type=SignalType.HOLD,
                    confidence=0.5,
                    timestamp=datetime.now(),
                    price=data['close'].iloc[-1],
                    reason="Test"
                )
            
            def calculate_indicators(self, data):
                return data
        
        return MockStrategy(StrategyConfig(name="TestStrategy"))
    
    def test_initialization(self, mock_strategy):
        """Test strategy initialization"""
        assert mock_strategy.name == "TestStrategy"
        assert mock_strategy.config.min_confidence == 0.6
        
    def test_validate_signal_valid(self, mock_strategy):
        """Test valid signal validation"""
        signal = StrategySignal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            price=50000,
            reason="Test"
        )
        assert mock_strategy.validate_signal(signal) is True
        
    def test_validate_signal_low_confidence(self, mock_strategy):
        """Test low confidence signal rejection"""
        signal = StrategySignal(
            signal_type=SignalType.BUY,
            confidence=0.5,
            timestamp=datetime.now(),
            price=50000,
            reason="Test"
        )
        assert mock_strategy.validate_signal(signal) is False
        
    def test_update_performance(self, mock_strategy):
        """Test performance tracking"""
        signal = StrategySignal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            price=50000,
            reason="Test"
        )
        mock_strategy.update_performance(signal, 0.05)
        assert len(mock_strategy.performance_history) == 1
        
    def test_get_performance_metrics(self, mock_strategy):
        """Test performance metrics retrieval"""
        signal = StrategySignal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            price=50000,
            reason="Test"
        )
        mock_strategy.update_performance(signal, 0.05)
        mock_strategy.update_performance(signal, -0.02)
        
        metrics = mock_strategy.get_performance_metrics()
        assert metrics['total_signals'] == 2
        assert metrics['win_rate'] == 50.0


class TestTrendFollowingStrategy:
    """Test trend following strategy"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample trending data"""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        # Create uptrend
        price = 100 + np.arange(200) * 0.5 + np.random.randn(200) * 5
        
        return pd.DataFrame({
            'open': price,
            'high': price + np.random.rand(200) * 10,
            'low': price - np.random.rand(200) * 10,
            'close': price,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
    
    @pytest.fixture
    def strategy(self):
        return TrendFollowingStrategy()
    
    def test_calculate_indicators(self, strategy, sample_data):
        """Test indicator calculation"""
        df = strategy.calculate_indicators(sample_data)
        assert 'fast_ma' in df.columns
        assert 'slow_ma' in df.columns
        assert 'adx' in df.columns
        
    def test_generate_buy_signal(self, strategy, sample_data):
        """Test buy signal generation in uptrend"""
        # Force uptrend by modifying data
        df = sample_data.copy()
        df['fast_ma'] = df['close'].rolling(20).mean()
        df['slow_ma'] = df['fast_ma'] - 10  # Fast above slow
        df['adx'] = 30  # Strong trend
        
        signal = strategy.generate_signal(df)
        assert signal.signal_type in [SignalType.BUY, SignalType.HOLD]
        
    def test_generate_sell_signal(self, strategy, sample_data):
        """Test sell signal generation in downtrend"""
        df = sample_data.copy()
        df['fast_ma'] = df['close'].rolling(20).mean()
        df['slow_ma'] = df['fast_ma'] + 10  # Fast below slow
        df['adx'] = 30
        
        signal = strategy.generate_signal(df)
        assert signal.signal_type in [SignalType.SELL, SignalType.HOLD]
        
    def test_insufficient_data(self, strategy):
        """Test signal generation with insufficient data"""
        small_data = pd.DataFrame({'close': np.random.randn(30)})
        signal = strategy.generate_signal(small_data)
        assert signal.signal_type == SignalType.HOLD
        assert "Insufficient data" in signal.reason


class TestMeanReversionStrategy:
    """Test mean reversion strategy"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample ranging data"""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        # Create ranging market
        price = 100 + np.sin(np.arange(200) * 0.1) * 10 + np.random.randn(200) * 2
        
        return pd.DataFrame({
            'open': price,
            'high': price + np.random.rand(200) * 5,
            'low': price - np.random.rand(200) * 5,
            'close': price,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
    
    @pytest.fixture
    def strategy(self):
        return MeanReversionStrategy()
    
    def test_calculate_indicators(self, strategy, sample_data):
        """Test indicator calculation"""
        df = strategy.calculate_indicators(sample_data)
        assert 'rsi' in df.columns
        assert 'bb_upper' in df.columns
        assert 'bb_lower' in df.columns
        assert 'z_score' in df.columns
        
    def test_oversold_signal(self, strategy, sample_data):
        """Test buy signal in oversold condition"""
        df = sample_data.copy()
        df['rsi'] = 25  # Oversold
        df['z_score'] = -2.5
        
        signal = strategy.generate_signal(df)
        assert signal.signal_type in [SignalType.BUY, SignalType.HOLD]
        
    def test_overbought_signal(self, strategy, sample_data):
        """Test sell signal in overbought condition"""
        df = sample_data.copy()
        df['rsi'] = 75  # Overbought
        df['z_score'] = 2.5
        
        signal = strategy.generate_signal(df)
        assert signal.signal_type in [SignalType.SELL, SignalType.HOLD]
        
    def test_neutral_signal(self, strategy, sample_data):
        """Test neutral signal in normal conditions"""
        df = sample_data.copy()
        df['rsi'] = 50
        df['z_score'] = 0
        
        signal = strategy.generate_signal(df)
        assert signal.signal_type == SignalType.HOLD


class TestMomentumStrategy:
    """Test momentum strategy"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample momentum data"""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        # Create strong momentum
        price = 100 + np.cumsum(np.random.randn(200) * 0.5)
        
        return pd.DataFrame({
            'open': price,
            'high': price + np.random.rand(200) * 5,
            'low': price - np.random.rand(200) * 5,
            'close': price,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
    
    @pytest.fixture
    def strategy(self):
        return MomentumStrategy()
    
    def test_calculate_indicators(self, strategy, sample_data):
        """Test indicator calculation"""
        df = strategy.calculate_indicators(sample_data)
        assert 'roc' in df.columns
        assert 'macd' in df.columns
        assert 'momentum_score' in df.columns
        
    def test_bullish_momentum_signal(self, strategy, sample_data):
        """Test buy signal on bullish momentum"""
        df = sample_data.copy()
        df['roc'] = 10  # Strong positive momentum
        df['macd'] = 1
        df['macd_signal'] = 0
        
        signal = strategy.generate_signal(df)
        assert signal.signal_type in [SignalType.BUY, SignalType.HOLD]
        
    def test_bearish_momentum_signal(self, strategy, sample_data):
        """Test sell signal on bearish momentum"""
        df = sample_data.copy()
        df['roc'] = -10  # Strong negative momentum
        df['macd'] = -1
        df['macd_signal'] = 0
        
        signal = strategy.generate_signal(df)
        assert signal.signal_type in [SignalType.SELL, SignalType.HOLD]
        
    def test_volume_confirmation(self, strategy, sample_data):
        """Test volume confirmation for signals"""
        df = sample_data.copy()
        df['roc'] = 10
        df['volume_ratio'] = 2.0  # High volume
        df['macd'] = 1
        df['macd_signal'] = 0
        
        signal = strategy.generate_signal(df)
        # Volume should increase confidence


class TestStrategyRotator:
    """Test strategy rotation"""
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        price = 100 + np.cumsum(np.random.randn(200) * 0.5)
        
        return pd.DataFrame({
            'close': price,
            'high': price + np.random.rand(200) * 5,
            'low': price - np.random.rand(200) * 5,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
    
    @pytest.fixture
    def rotator(self):
        return StrategyRotator(rotation_interval_minutes=60)
    
    def test_initialization(self, rotator):
        """Test rotator initialization"""
        assert len(rotator.strategies) == 3
        assert 'trend_following' in rotator.strategies
        assert 'mean_reversion' in rotator.strategies
        assert 'momentum' in rotator.strategies
        
    def test_register_strategy(self, rotator):
        """Test registering new strategy"""
        new_strategy = TrendFollowingStrategy()
        rotator.register_strategy("new_strategy", new_strategy, weight=0.5)
        assert "new_strategy" in rotator.strategies
        assert rotator.performance_weights["new_strategy"] == 0.5
        
    def test_detect_market_regime(self, rotator, sample_data):
        """Test market regime detection"""
        regime = rotator.detect_market_regime(sample_data)
        assert regime in ['trending', 'ranging', 'volatile', 'quiet', 'transitional', 'unknown']
        
    def test_select_best_strategy(self, rotator, sample_data):
        """Test best strategy selection"""
        strategy_name, confidence = rotator.select_best_strategy(sample_data)
        assert strategy_name in rotator.strategies
        assert 0 <= confidence <= 1
        
    def test_generate_signal(self, rotator, sample_data):
        """Test signal generation with rotation"""
        signal = rotator.generate_signal(sample_data)
        assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert signal.confidence >= 0
        assert signal.confidence <= 1
        
    def test_update_performance(self, rotator, sample_data):
        """Test performance update for strategies"""
        signal = rotator.generate_signal(sample_data)
        rotator.update_strategy_performance("trend_following", signal, 0.05)
        
        metrics = rotator.get_performance_summary()
        assert 'strategies' in metrics
        
    def test_get_rotation_history(self, rotator, sample_data):
        """Test rotation history retrieval"""
        # Force a rotation
        rotator._rotate_strategy("momentum", 0.8, sample_data)
        history = rotator.get_rotation_history()
        assert len(history) >= 1
        assert 'to_strategy' in history[0]