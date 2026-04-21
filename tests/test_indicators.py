"""
test_indicators.py
Unit tests for technical indicators.
"""

import pytest
import pandas as pd
import numpy as np

from app.analysis.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """Test technical indicator calculations"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        
        price = 100 + np.cumsum(np.random.randn(200) * 0.5)
        
        df = pd.DataFrame({
            'open': price,
            'high': price * (1 + np.random.rand(200) * 0.01),
            'low': price * (1 - np.random.rand(200) * 0.01),
            'close': price,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def indicators(self, sample_data):
        """Create TechnicalIndicators instance"""
        return TechnicalIndicators(sample_data)
    
    def test_initialization(self, indicators, sample_data):
        """Test initialization"""
        assert indicators.data is not None
        assert len(indicators.data) == len(sample_data)
        
    def test_sma_calculation(self, indicators):
        """Test SMA calculation"""
        sma = indicators.sma(period=20)
        assert len(sma) == len(indicators.data)
        assert sma.isna().sum() == 19  # First 19 periods are NaN
        
    def test_ema_calculation(self, indicators):
        """Test EMA calculation"""
        ema = indicators.ema(period=20)
        assert len(ema) == len(indicators.data)
        
    def test_rsi_calculation(self, indicators):
        """Test RSI calculation"""
        rsi = indicators.rsi(period=14)
        assert len(rsi) == len(indicators.data)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
        
    def test_macd_calculation(self, indicators):
        """Test MACD calculation"""
        macd, signal, hist = indicators.macd()
        assert len(macd) == len(indicators.data)
        assert len(signal) == len(indicators.data)
        assert len(hist) == len(indicators.data)
        
    def test_bollinger_bands(self, indicators):
        """Test Bollinger Bands calculation"""
        upper, middle, lower = indicators.bollinger_bands()
        assert len(upper) == len(indicators.data)
        assert len(middle) == len(indicators.data)
        assert len(lower) == len(indicators.data)
        # Upper should be >= middle >= lower
        valid_idx = ~upper.isna()
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()
        
    def test_atr_calculation(self, indicators):
        """Test ATR calculation"""
        atr = indicators.atr(period=14)
        assert len(atr) == len(indicators.data)
        assert (atr.dropna() >= 0).all()
        
    def test_stochastic(self, indicators):
        """Test Stochastic Oscillator"""
        k, d = indicators.stochastic()
        assert len(k) == len(indicators.data)
        assert len(d) == len(indicators.data)
        
    def test_williams_r(self, indicators):
        """Test Williams %R"""
        williams = indicators.williams_r()
        assert len(williams) == len(indicators.data)
        valid_williams = williams.dropna()
        assert (valid_williams >= -100).all()
        assert (valid_williams <= 0).all()
        
    def test_cci(self, indicators):
        """Test CCI calculation"""
        cci = indicators.cci(period=20)
        assert len(cci) == len(indicators.data)
        
    def test_adx(self, indicators):
        """Test ADX calculation"""
        adx = indicators.adx(period=14)
        assert len(adx) == len(indicators.data)
        assert (adx.dropna() >= 0).all()
        
    def test_mfi(self, indicators):
        """Test MFI calculation"""
        mfi = indicators.mfi(period=14)
        assert len(mfi) == len(indicators.data)
        valid_mfi = mfi.dropna()
        assert (valid_mfi >= 0).all()
        assert (valid_mfi <= 100).all()
        
    def test_obv(self, indicators):
        """Test OBV calculation"""
        obv = indicators.obv()
        assert len(obv) == len(indicators.data)
        
    def test_volume_sma(self, indicators):
        """Test Volume SMA"""
        vol_sma = indicators.volume_sma(period=20)
        assert len(vol_sma) == len(indicators.data)
        
    def test_ichimoku(self, indicators):
        """Test Ichimoku Cloud"""
        ichimoku = indicators.ichimoku()
        assert 'conversion_line' in ichimoku
        assert 'base_line' in ichimoku
        assert 'leading_span_a' in ichimoku
        assert 'leading_span_b' in ichimoku
        
    def test_calculate_all(self, indicators):
        """Test calculating all indicators"""
        results = indicators.calculate_all()
        assert isinstance(results, dict)
        assert len(results) > 0
        
    def test_get_current_values(self, indicators):
        """Test getting current indicator values"""
        indicators.calculate_all()
        current = indicators.get_current_values()
        assert isinstance(current, dict)
        assert len(current) > 0


class TestIndicatorEdgeCases:
    """Test edge cases for indicators"""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame()
        indicators = TechnicalIndicators(df)
        
        with pytest.raises(Exception):
            indicators.sma(20)
            
    def test_insufficient_data(self):
        """Test with insufficient data"""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1H')
        df = pd.DataFrame({
            'open': np.random.randn(10),
            'high': np.random.randn(10),
            'low': np.random.randn(10),
            'close': np.random.randn(10),
            'volume': np.random.randint(100, 1000, 10)
        }, index=dates)
        
        indicators = TechnicalIndicators(df)
        rsi = indicators.rsi(period=14)
        # Should have many NaN values
        assert rsi.isna().sum() >= 10
        
    def test_constant_price(self):
        """Test with constant price (no movement)"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        df = pd.DataFrame({
            'open': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        }, index=dates)
        
        indicators = TechnicalIndicators(df)
        rsi = indicators.rsi(period=14).dropna()
        # RSI should be around 50 for constant price
        assert abs(rsi.iloc[-1] - 50) < 10