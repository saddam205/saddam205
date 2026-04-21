"""
test_risk.py
Unit tests for risk management.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from app.core.risk_manager import RiskManager
from app.filters.trade_filter import TradeFilter, TradeSignal, FilterCondition


class TestRiskManager:
    """Test risk management functionality"""
    
    @pytest.fixture
    def risk_manager(self):
        return RiskManager()
    
    def test_initialization(self, risk_manager):
        """Test risk manager initialization"""
        assert risk_manager.limits['max_position_pct'] == 0.10
        assert risk_manager.consecutive_losses == 0
        assert risk_manager.current_drawdown == 0
        
    def test_check_position_valid(self, risk_manager):
        """Test position validation for valid position"""
        valid, reason = risk_manager.check_position(
            symbol="BTCUSDT",
            size=0.1,
            price=50000,
            capital=100000
        )
        assert valid is True
        assert reason == "OK"
        
    def test_check_position_exceeds_limit(self, risk_manager):
        """Test position validation exceeding limit"""
        valid, reason = risk_manager.check_position(
            symbol="BTCUSDT",
            size=2.0,  # $100,000 position
            price=50000,
            capital=100000
        )
        assert valid is False
        assert "exceeds limit" in reason
        
    def test_update_pnl_winning_trade(self, risk_manager):
        """Test P&L update for winning trade"""
        risk_manager._update_pnl(1000)
        assert risk_manager.consecutive_losses == 0
        
    def test_update_pnl_losing_trade(self, risk_manager):
        """Test P&L update for losing trade"""
        risk_manager._update_pnl(-500)
        assert risk_manager.consecutive_losses == 1
        
    def test_consecutive_losses_limit(self, risk_manager):
        """Test consecutive losses limit enforcement"""
        for _ in range(5):
            risk_manager._update_pnl(-100)
        
        valid, reason = risk_manager.check_position(
            symbol="BTCUSDT",
            size=0.1,
            price=50000,
            capital=100000
        )
        assert valid is False
        assert "Consecutive losses" in reason
        
    def test_calculate_var(self, risk_manager):
        """Test VaR calculation"""
        returns = np.random.randn(100) * 0.02
        var = risk_manager.calculate_var(returns, confidence=0.95)
        assert var < 0  # VaR should be negative
        assert var > -0.1  # Reasonable range
        
    def test_calculate_cvar(self, risk_manager):
        """Test CVaR calculation"""
        returns = np.random.randn(100) * 0.02
        cvar = risk_manager.calculate_cvar(returns, confidence=0.95)
        assert cvar < 0
        assert cvar >= risk_manager.calculate_var(returns)  # CVaR <= VaR
        
    def test_daily_pnl_tracking(self, risk_manager):
        """Test daily P&L tracking"""
        risk_manager._update_pnl(100)
        risk_manager._update_pnl(200)
        
        assert len(risk_manager.daily_pnl) == 1
        assert risk_manager.daily_pnl[-1]['pnl'] == 300
        
    def test_daily_loss_limit(self, risk_manager):
        """Test daily loss limit enforcement"""
        # Simulate large daily loss
        risk_manager.daily_pnl = [{'date': datetime.now().date(), 'pnl': -6000}]
        risk_manager.limits['max_daily_loss'] = 0.05  # 5% of $100k = $5000
        
        valid, reason = risk_manager.check_position(
            symbol="BTCUSDT",
            size=0.1,
            price=50000,
            capital=100000
        )
        assert valid is False
        assert "Daily loss limit" in reason
        
    def test_get_risk_report(self, risk_manager):
        """Test risk report generation"""
        report = risk_manager.get_risk_report()
        assert 'current_drawdown' in report
        assert 'max_drawdown' in report
        assert 'limits' in report
        
    def test_update_limits(self, risk_manager):
        """Test updating risk limits"""
        new_limits = {'max_position_pct': 0.15, 'max_daily_loss': 0.08}
        updated = risk_manager.update_limits(new_limits)
        
        assert updated['max_position_pct'] == 0.15
        assert updated['max_daily_loss'] == 0.08


class TestTradeFilter:
    """Test trade filtering"""
    
    @pytest.fixture
    def trade_filter(self):
        return TradeFilter()
    
    @pytest.fixture
    def sample_signal(self):
        return TradeSignal(
            symbol="BTCUSDT",
            signal="BUY",
            confidence=0.8,
            price=50000,
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def sample_market_data(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        return pd.DataFrame({
            'close': 50000 + np.cumsum(np.random.randn(100) * 100),
            'high': 51000 + np.random.randn(100) * 100,
            'low': 49000 + np.random.randn(100) * 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    @pytest.fixture
    def sample_portfolio(self):
        return {'open_positions': 0, 'today_trades': 0, 'current_drawdown': 0}
    
    def test_confidence_filter_passes(self, trade_filter, sample_signal, 
                                       sample_market_data, sample_portfolio):
        """Test confidence filter passes for high confidence"""
        passed, results = trade_filter.filter_signal(
            sample_signal, sample_market_data, sample_portfolio
        )
        assert passed is True
        
    def test_confidence_filter_fails(self, trade_filter, sample_signal,
                                      sample_market_data, sample_portfolio):
        """Test confidence filter fails for low confidence"""
        sample_signal.confidence = 0.5
        trade_filter.config['min_confidence'] = 0.65
        
        passed, results = trade_filter.filter_signal(
            sample_signal, sample_market_data, sample_portfolio
        )
        assert passed is False
        assert results[0].condition == FilterCondition.MIN_CONFIDENCE
        
    def test_register_custom_filter(self, trade_filter):
        """Test registering custom filter"""
        def custom_filter(signal, market_data, portfolio):
            from app.filters.trade_filter import FilterResult
            return FilterResult(
                passed=True,
                condition=FilterCondition.CUSTOM,
                reason="Custom filter passed"
            )
        
        trade_filter.register_filter(FilterCondition.CUSTOM, custom_filter)
        assert len(trade_filter.filters) > 0
        
    def test_get_statistics(self, trade_filter, sample_signal,
                           sample_market_data, sample_portfolio):
        """Test statistics collection"""
        trade_filter.filter_signal(sample_signal, sample_market_data, sample_portfolio)
        stats = trade_filter.get_statistics()
        
        assert stats['total_signals'] == 1
        assert 'pass_rate' in stats
        
    def test_update_config(self, trade_filter):
        """Test configuration update"""
        trade_filter.update_config({'min_confidence': 0.75})
        assert trade_filter.config['min_confidence'] == 0.75