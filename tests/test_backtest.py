"""
test_backtest.py
Unit tests for backtesting engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.validation.backtest_engine import BacktestEngine, CostAdjustedBacktest, RealisticCostCalculator
from app.validation.walk_forward_validator import WalkForwardValidator
from app.validation.monte_carlo_simulator import MonteCarloSimulator


class TestBacktestEngine:
    """Test backtest engine functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        np.random.seed(42)
        
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
    def backtest_engine(self):
        """Create backtest engine instance"""
        return BacktestEngine(initial_capital=100000, commission=0.001)
    
    def test_initialization(self, backtest_engine):
        """Test engine initialization"""
        assert backtest_engine.initial_capital == 100000
        assert backtest_engine.capital == 100000
        assert backtest_engine.commission == 0.001
        assert backtest_engine.positions == []
        assert backtest_engine.trades == []
        
    def test_open_position(self, backtest_engine, sample_data):
        """Test opening a position"""
        price = sample_data['close'].iloc[0]
        backtest_engine._open_position(sample_data.index[0], price, 'BUY')
        
        assert len(backtest_engine.positions) == 1
        assert backtest_engine.capital < 100000
        
    def test_close_position(self, backtest_engine, sample_data):
        """Test closing a position"""
        price = sample_data['close'].iloc[0]
        backtest_engine._open_position(sample_data.index[0], price, 'BUY')
        
        exit_price = sample_data['close'].iloc[10]
        backtest_engine._close_position(sample_data.index[10], exit_price, 'SELL')
        
        assert len(backtest_engine.positions) == 0
        assert len(backtest_engine.trades) == 1
        
    def test_run_backtest_no_trades(self, backtest_engine, sample_data):
        """Test backtest with no signals"""
        def strategy(data, **params):
            return pd.Series(0, index=data.index)
        
        metrics, trades, equity = backtest_engine.run_backtest(sample_data, strategy)
        
        assert metrics['total_trades'] == 0
        assert metrics['total_return'] == 0
        
    def test_run_backtest_with_signals(self, backtest_engine, sample_data):
        """Test backtest with buy/sell signals"""
        def strategy(data, **params):
            signals = pd.Series(0, index=data.index)
            # Buy on first day, sell on last day
            signals.iloc[0] = 1
            signals.iloc[-1] = -1
            return signals
        
        metrics, trades, equity = backtest_engine.run_backtest(sample_data, strategy)
        
        assert len(trades) >= 1
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
    def test_calculate_metrics_no_trades(self, backtest_engine):
        """Test metrics calculation with no trades"""
        metrics = backtest_engine.calculate_metrics()
        assert 'error' in metrics
        
    def test_commission_impact(self, sample_data):
        """Test that commission affects returns"""
        engine1 = BacktestEngine(initial_capital=100000, commission=0)
        engine2 = BacktestEngine(initial_capital=100000, commission=0.01)
        
        def strategy(data, **params):
            signals = pd.Series(0, index=data.index)
            signals.iloc[0] = 1
            signals.iloc[-1] = -1
            return signals
        
        metrics1, _, _ = engine1.run_backtest(sample_data, strategy)
        metrics2, _, _ = engine2.run_backtest(sample_data, strategy)
        
        # Higher commission should result in lower return
        assert metrics2['total_return'] <= metrics1['total_return']
        
    def test_equity_curve_recording(self, backtest_engine, sample_data):
        """Test equity curve is recorded"""
        def strategy(data, **params):
            signals = pd.Series(0, index=data.index)
            signals.iloc[0] = 1
            signals.iloc[-1] = -1
            return signals
        
        _, _, equity = backtest_engine.run_backtest(sample_data, strategy)
        
        assert len(equity) == len(sample_data)
        assert 'timestamp' in equity[0]
        assert 'equity' in equity[0]


class TestCostAdjustedBacktest:
    """Test realistic cost-adjusted backtesting"""
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        price = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        return pd.DataFrame({
            'open': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'close': price,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_cost_calculator(self):
        """Test realistic cost calculator"""
        calculator = RealisticCostCalculator()
        
        costs = calculator.calculate_trade_cost(
            trade_size=1.0,
            price=50000,
            volatility=0.02,
            volume=1000000,
            is_market_order=True
        )
        
        assert 'exchange_fee' in costs
        assert 'slippage' in costs
        assert 'spread' in costs
        assert 'total_cost' in costs
        assert costs['total_cost'] > 0
        
    def test_slippage_increases_with_size(self):
        """Test that slippage increases with order size"""
        calculator = RealisticCostCalculator()
        
        small_order = calculator.calculate_trade_cost(1.0, 50000, 0.02, 1000000)
        large_order = calculator.calculate_trade_cost(100.0, 50000, 0.02, 1000000)
        
        assert large_order['slippage'] >= small_order['slippage']


class TestWalkForwardValidator:
    """Test walk-forward validation"""
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2024-01-01', periods=500, freq='1H')
        price = 100 + np.cumsum(np.random.randn(500) * 0.5)
        
        return pd.DataFrame({
            'close': price,
            'high': price * 1.01,
            'low': price * 0.99,
            'volume': np.random.randint(1000, 10000, 500)
        }, index=dates)
    
    class MockModel:
        def fit(self, train, val):
            pass
        def evaluate(self, test):
            return {
                'accuracy': 0.58,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.08,
                'win_rate': 0.62,
                'total_trades': 45
            }
    
    def test_initialization(self):
        """Test validator initialization"""
        validator = WalkForwardValidator(self.MockModel)
        assert validator.train_ratio == 0.6
        assert validator.val_ratio == 0.2
        assert validator.test_ratio == 0.2
        
    def test_run_validation(self, sample_data):
        """Test running walk-forward validation"""
        validator = WalkForwardValidator(self.MockModel)
        results = validator.run_validation(sample_data, n_splits=3)
        
        assert len(results) > 0
        assert 'test_accuracy' in results[0]
        
    def test_aggregate_metrics(self, sample_data):
        """Test aggregate metrics calculation"""
        validator = WalkForwardValidator(self.MockModel)
        validator.run_validation(sample_data, n_splits=3)
        
        assert hasattr(validator, 'aggregate')
        assert 'mean_accuracy' in validator.aggregate
        assert 'robustness_score' in validator.aggregate


class TestMonteCarloSimulator:
    """Test Monte Carlo simulation"""
    
    def test_initialization(self):
        """Test simulator initialization"""
        simulator = MonteCarloSimulator(n_simulations=100, n_days=50)
        assert simulator.n_simulations == 100
        assert simulator.n_days == 50
        
    def test_simulate_from_returns(self):
        """Test simulation from historical returns"""
        returns = np.random.randn(252) * 0.02
        simulator = MonteCarloSimulator(n_simulations=100, n_days=50)
        
        result = simulator.simulate_from_returns(returns, initial_capital=100000)
        
        assert result.mean_final_value > 0
        assert result.median_final_value > 0
        assert result.var_95 is not None
        assert result.cvar_95 is not None
        
    def test_simulate_normal(self):
        """Test normal distribution simulation"""
        simulator = MonteCarloSimulator(n_simulations=100, n_days=50)
        result = simulator.simulate_normal(mu=0.0001, sigma=0.02, initial_capital=100000)
        
        assert result.mean_final_value > 0
        assert result.success_rate >= 0
        assert result.success_rate <= 100
        
    def test_get_confidence_interval(self, sample_returns):
        """Test confidence interval calculation"""
        simulator = MonteCarloSimulator(n_simulations=100, n_days=50)
        simulator.simulate_from_returns(sample_returns)
        
        lower, upper = simulator.get_confidence_interval(confidence=0.95)
        assert lower < upper
        
    @pytest.fixture
    def sample_returns(self):
        return np.random.randn(252) * 0.02