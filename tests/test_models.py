"""
test_models.py
Unit tests for AI/ML models.
"""

import pytest
import numpy as np
import pandas as pd

from app.models.xgboost_model import XGBoostModel
from app.models.ensemble import ModelEnsemble
from app.models.position_sizer import DynamicPositionSizer
from app.models.indicator_selector import AutoIndicatorSelector


class TestXGBoostModel:
    """Test XGBoost model"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample feature data"""
        np.random.seed(42)
        X = np.random.randn(500, 20)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y
    
    @pytest.fixture
    def model(self):
        return XGBoostModel()
    
    def test_initialization(self, model):
        """Test model initialization"""
        assert model.model is None
        assert model.best_score == float('inf')
        
    def test_train_and_predict(self, model, sample_data):
        """Test training and prediction"""
        X, y = sample_data
        metrics = model.train(X, y, validation_split=0.2)
        
        assert 'validation_accuracy' in metrics
        assert model.model is not None
        
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
        assert set(predictions).issubset({0, 1})
        
    def test_predict_proba(self, model, sample_data):
        """Test probability prediction"""
        X, y = sample_data
        model.train(X, y)
        
        proba = model.predict_proba(X[:5])
        assert proba.shape[0] == 5
        assert proba.shape[1] == 2
        
    def test_predict_with_confidence(self, model, sample_data):
        """Test prediction with confidence"""
        X, y = sample_data
        model.train(X, y)
        
        results = model.predict_with_confidence(X[:5])
        assert len(results) == 5
        for result in results:
            assert 'signal' in result
            assert 'confidence' in result
            assert 0 <= result['confidence'] <= 1
            
    def test_feature_importance(self, model, sample_data):
        """Test feature importance extraction"""
        X, y = sample_data
        model.train(X, y)
        
        importance = model.get_feature_importance(top_n=5)
        assert len(importance) <= 5
        assert isinstance(importance[0], tuple)
        
    def test_cross_validate(self, model, sample_data):
        """Test cross-validation"""
        X, y = sample_data
        results = model.cross_validate(X, y, n_splits=3)
        
        assert 'mean_accuracy' in results
        assert 'mean_sharpe' in results
        assert 0 <= results['mean_accuracy'] <= 1


class TestModelEnsemble:
    """Test ensemble model"""
    
    @pytest.fixture
    def ensemble(self):
        return ModelEnsemble()
    
    @pytest.fixture
    def mock_models(self):
        class MockModel:
            def predict_with_confidence(self, X):
                return {'signal': 'BUY', 'confidence': 0.7}
        
        return {
            'model1': MockModel(),
            'model2': MockModel(),
            'model3': MockModel()
        }
    
    def test_add_model(self, ensemble, mock_models):
        """Test adding models to ensemble"""
        ensemble.add_model('test', mock_models['model1'], initial_weight=1.0)
        assert 'test' in ensemble.models
        assert ensemble.weights['test'].weight == 1.0
        
    def test_predict_weighted_voting(self, ensemble, mock_models):
        """Test weighted voting prediction"""
        for name, model in mock_models.items():
            ensemble.add_model(name, model, initial_weight=1.0)
        
        X = np.random.randn(10, 5)
        result = ensemble.predict(X, method='weighted_voting')
        
        assert 'signal' in result
        assert 'confidence' in result
        assert result['ensemble_size'] == 3
        
    def test_predict_average(self, ensemble, mock_models):
        """Test average prediction"""
        for name, model in mock_models.items():
            ensemble.add_model(name, model, initial_weight=1.0)
        
        X = np.random.randn(10, 5)
        result = ensemble.predict(X, method='average')
        
        assert 'signal' in result
        assert 'confidence' in result
        
    def test_update_weights(self, ensemble, mock_models):
        """Test dynamic weight updating"""
        for name, model in mock_models.items():
            ensemble.add_model(name, model, initial_weight=1.0)
        
        # Simulate predictions
        X = np.random.randn(10, 5)
        ensemble.predict(X)
        
        # Update weights based on outcomes
        outcomes = [
            {'timestamp': ensemble.prediction_history[0]['timestamp'], 
             'actual_signal': 'BUY'}
        ]
        ensemble.update_weights(outcomes)
        
        weights = ensemble.get_ensemble_weights()
        assert isinstance(weights, dict)
        
    def test_get_performance_summary(self, ensemble, mock_models):
        """Test performance summary"""
        for name, model in mock_models.items():
            ensemble.add_model(name, model, initial_weight=1.0)
        
        summary = ensemble.get_performance_summary()
        assert 'models' in summary
        assert 'weights' in summary


class TestDynamicPositionSizer:
    """Test position sizing"""
    
    @pytest.fixture
    def sizer(self):
        return DynamicPositionSizer(max_position_pct=0.25, min_position_pct=0.01)
    
    def test_calculate_position(self, sizer):
        """Test position size calculation"""
        position_value, quantity, info = sizer.calculate_position(
            balance=100000,
            confidence=0.8,
            volatility=0.02,
            current_price=50000
        )
        
        assert position_value > 0
        assert position_value <= 25000  # 25% max
        assert quantity > 0
        assert 'final_size' in info
        
    def test_calculate_position_by_investment(self, sizer):
        """Test position calculation by investment amount"""
        position_value, reason = sizer.calculate_position_by_investment(
            investment_amount=5000,
            balance=100000,
            confidence=0.7
        )
        
        assert position_value == 5000
        assert "accepted" in reason.lower()
        
    def test_max_position_limit(self, sizer):
        """Test max position limit enforcement"""
        position_value, reason = sizer.calculate_position_by_investment(
            investment_amount=50000,  # 50% of balance
            balance=100000,
            confidence=0.7
        )
        
        assert position_value <= 25000  # Max 25%
        
    def test_min_position_limit(self, sizer):
        """Test min position limit enforcement"""
        position_value, reason = sizer.calculate_position_by_investment(
            investment_amount=1,
            balance=100000,
            confidence=0.7
        )
        
        assert position_value >= 1000  # 1% of 100k
        
    def test_stop_loss_calculation(self, sizer):
        """Test stop loss calculation"""
        stop_loss = sizer.calculate_stop_loss(
            entry_price=50000,
            confidence=0.8,
            volatility=0.02,
            stop_loss_type='atr'
        )
        
        assert stop_loss < 50000
        assert stop_loss > 0
        
    def test_take_profit_calculation(self, sizer):
        """Test take profit calculation"""
        stop_loss = 49000
        take_profit = sizer.calculate_take_profit(
            entry_price=50000,
            stop_loss=stop_loss,
            risk_reward_ratio=2.0
        )
        
        assert take_profit > 50000
        expected = 50000 + (50000 - stop_loss) * 2
        assert abs(take_profit - expected) < 1
        
    def test_kelly_parameters_update(self, sizer):
        """Test Kelly parameter update"""
        sizer.update_kelly_parameters(win_rate=0.6, avg_win_loss_ratio=1.5)
        assert len(sizer.win_rate_history) == 1


class TestAutoIndicatorSelector:
    """Test automatic indicator selection"""
    
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
    def selector(self):
        return AutoIndicatorSelector()
    
    def test_select_best_indicators(self, selector, sample_data):
        """Test indicator selection"""
        indicators = selector.select_best_indicators(sample_data)
        assert len(indicators) >= 5
        assert isinstance(indicators, list)
        
    def test_default_indicators_fallback(self, selector):
        """Test default indicators fallback for insufficient data"""
        small_data = pd.DataFrame({'close': np.random.randn(50)})
        indicators = selector.select_best_indicators(small_data)
        assert indicators == selector.default_indicators
        
    def test_market_regime_detection(self, selector, sample_data):
        """Test market regime detection"""
        regime = selector._detect_market_regime(sample_data)
        assert regime in ['trending_up', 'trending_down', 'high_volatility', 
                         'low_volatility', 'ranging', 'mixed']
        
    def test_get_performance(self, selector, sample_data):
        """Test performance metrics retrieval"""
        selector.select_best_indicators(sample_data)
        performance = selector.get_indicator_performance()
        assert isinstance(performance, dict)