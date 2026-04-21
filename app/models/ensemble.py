"""
ensemble.py
Part of the app/models module.
Ensemble model combining multiple AI models for robust predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelWeight:
    """Weight configuration for ensemble models"""
    model_name: str
    weight: float
    performance_score: float = 1.0
    last_updated: datetime = None


class ModelEnsemble:
    """
    Ensemble model that combines predictions from multiple AI models.
    Supports weighted voting, stacking, and dynamic weight adjustment.
    """
    
    def __init__(self, dynamic_weights: bool = True, 
                 weight_decay: float = 0.95):
        """
        Initialize ensemble
        
        Args:
            dynamic_weights: Whether to adjust weights based on performance
            weight_decay: Decay factor for historical performance
        """
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, ModelWeight] = {}
        self.dynamic_weights = dynamic_weights
        self.weight_decay = weight_decay
        self.prediction_history: List[Dict] = []
        self.performance_tracker: Dict[str, List[float]] = {}
        
    def add_model(self, name: str, model: Any, initial_weight: float = 1.0):
        """
        Add a model to the ensemble
        
        Args:
            name: Model identifier
            model: Model object with predict method
            initial_weight: Initial weight for this model
        """
        self.models[name] = model
        self.weights[name] = ModelWeight(
            model_name=name,
            weight=initial_weight,
            performance_score=1.0,
            last_updated=datetime.now()
        )
        self.performance_tracker[name] = []
        
        logger.info(f"Added model {name} with weight {initial_weight}")
    
    def predict(self, X: np.ndarray, method: str = 'weighted_voting') -> Dict:
        """
        Make ensemble prediction
        
        Args:
            X: Feature matrix
            method: Ensemble method ('weighted_voting', 'average', 'stacking')
        
        Returns:
            Ensemble prediction with confidence
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        confidences = []
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_with_confidence'):
                    result = model.predict_with_confidence(X)
                    pred = result['signal']
                    conf = result['confidence']
                elif hasattr(model, 'predict'):
                    pred = model.predict(X)
                    conf = 0.7  # Default confidence
                else:
                    continue
                
                predictions.append({
                    'model': name,
                    'prediction': pred,
                    'confidence': conf,
                    'weight': self.weights[name].weight
                })
                confidences.append(conf)
                
            except Exception as e:
                logger.error(f"Model {name} prediction failed: {e}")
        
        if not predictions:
            return {'signal': 'HOLD', 'confidence': 0.5, 'ensemble_size': 0}
        
        # Apply ensemble method
        if method == 'weighted_voting':
            result = self._weighted_voting(predictions)
        elif method == 'average':
            result = self._average_predictions(predictions)
        else:
            result = self._stacking_predictions(predictions, X)
        
        result['ensemble_size'] = len(predictions)
        result['model_predictions'] = predictions
        
        # Store history
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'result': result,
            'predictions': predictions
        })
        
        # Keep only last 1000
        if len(self.prediction_history) > 1000:
            self.prediction_history.pop(0)
        
        return result
    
    def _weighted_voting(self, predictions: List[Dict]) -> Dict:
        """
        Weighted voting ensemble
        
        Args:
            predictions: List of model predictions
        
        Returns:
            Weighted voting result
        """
        signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_weight = 0
        
        for pred in predictions:
            signal = pred['prediction']
            weight = pred['weight'] * pred['confidence']
            total_weight += weight
            
            if signal in signal_scores:
                signal_scores[signal] += weight
        
        # Find best signal
        if total_weight > 0:
            for signal in signal_scores:
                signal_scores[signal] /= total_weight
        
        best_signal = max(signal_scores, key=signal_scores.get)
        confidence = signal_scores[best_signal]
        
        return {
            'signal': best_signal,
            'confidence': confidence,
            'signal_scores': signal_scores,
            'method': 'weighted_voting'
        }
    
    def _average_predictions(self, predictions: List[Dict]) -> Dict:
        """
        Average predictions ensemble
        
        Args:
            predictions: List of model predictions
        
        Returns:
            Averaged prediction
        """
        # Convert signals to numeric values
        signal_map = {'SELL': -1, 'HOLD': 0, 'BUY': 1}
        weighted_sum = 0
        total_weight = 0
        
        for pred in predictions:
            numeric_signal = signal_map.get(pred['prediction'], 0)
            weight = pred['weight'] * pred['confidence']
            weighted_sum += numeric_signal * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_signal = weighted_sum / total_weight
        else:
            avg_signal = 0
        
        # Convert back to signal
        if avg_signal > 0.3:
            signal = 'BUY'
            confidence = min(0.9, avg_signal)
        elif avg_signal < -0.3:
            signal = 'SELL'
            confidence = min(0.9, abs(avg_signal))
        else:
            signal = 'HOLD'
            confidence = 0.5
        
        return {
            'signal': signal,
            'confidence': confidence,
            'avg_signal_value': avg_signal,
            'method': 'average'
        }
    
    def _stacking_predictions(self, predictions: List[Dict], X: np.ndarray) -> Dict:
        """
        Stacking ensemble using meta-learner
        
        Args:
            predictions: List of model predictions
            X: Original features
        
        Returns:
            Stacked prediction
        """
        # Simple stacking: use highest confidence prediction
        best_pred = max(predictions, key=lambda x: x['weight'] * x['confidence'])
        
        return {
            'signal': best_pred['prediction'],
            'confidence': best_pred['confidence'] * best_pred['weight'],
            'selected_model': best_pred['model'],
            'method': 'stacking'
        }
    
    def update_weights(self, actual_outcomes: List[Dict]):
        """
        Update model weights based on actual performance
        
        Args:
            actual_outcomes: List of actual trading outcomes
        """
        if not self.dynamic_weights:
            return
        
        # Calculate performance for each model
        model_performance = {name: [] for name in self.models.keys()}
        
        for outcome in actual_outcomes:
            timestamp = outcome.get('timestamp')
            actual_signal = outcome.get('actual_signal')
            
            # Find corresponding predictions
            for history in self.prediction_history:
                if history['timestamp'] <= timestamp:
                    for pred in history['predictions']:
                        model_name = pred['model']
                        if pred['prediction'] == actual_signal:
                            model_performance[model_name].append(1.0)
                        else:
                            model_performance[model_name].append(0.0)
        
        # Update weights based on performance
        for name, performance in model_performance.items():
            if performance:
                avg_performance = np.mean(performance)
                self.performance_tracker[name].append(avg_performance)
                
                # Keep only last 100
                if len(self.performance_tracker[name]) > 100:
                    self.performance_tracker[name].pop(0)
                
                # Calculate rolling performance
                rolling_perf = np.mean(self.performance_tracker[name][-50:])
                
                # Update weight
                current_weight = self.weights[name].weight
                new_weight = current_weight * (1 + (rolling_perf - 0.5))
                
                # Apply decay to historical weights
                self.weights[name].weight = self.weight_decay * current_weight + (1 - self.weight_decay) * new_weight
                self.weights[name].performance_score = rolling_perf
                self.weights[name].last_updated = datetime.now()
                
                logger.debug(f"Updated weight for {name}: {current_weight:.3f} -> {self.weights[name].weight:.3f}")
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """
        Get current ensemble weights
        
        Returns:
            Dictionary of model weights
        """
        return {name: w.weight for name, w in self.weights.items()}
    
    def get_performance_summary(self) -> Dict:
        """
        Get ensemble performance summary
        
        Returns:
            Performance metrics
        """
        return {
            'models': list(self.models.keys()),
            'weights': self.get_ensemble_weights(),
            'dynamic_weights_enabled': self.dynamic_weights,
            'total_predictions': len(self.prediction_history),
            'model_performance': {
                name: {
                    'avg_score': np.mean(scores) if scores else 0,
                    'recent_score': np.mean(scores[-20:]) if scores else 0,
                    'samples': len(scores)
                }
                for name, scores in self.performance_tracker.items()
            }
        }
    
    def save(self, filepath: str):
        """
        Save ensemble configuration
        
        Args:
            filepath: Path to save ensemble
        """
        import joblib
        
        ensemble_data = {
            'weights': {name: w.__dict__ for name, w in self.weights.items()},
            'dynamic_weights': self.dynamic_weights,
            'performance_tracker': self.performance_tracker
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load ensemble configuration
        
        Args:
            filepath: Path to load ensemble from
        """
        import joblib
        
        ensemble_data = joblib.load(filepath)
        
        for name, weight_data in ensemble_data['weights'].items():
            if name in self.weights:
                self.weights[name].weight = weight_data['weight']
                self.weights[name].performance_score = weight_data['performance_score']
        
        self.dynamic_weights = ensemble_data['dynamic_weights']
        self.performance_tracker = ensemble_data['performance_tracker']
        
        logger.info(f"Ensemble loaded from {filepath}")


class EnsemblePredictor:
    """
    High-level ensemble predictor for live trading
    """
    
    def __init__(self):
        """Initialize ensemble predictor"""
        self.ensemble = ModelEnsemble()
        self.default_models = []
        
    def predict_with_ensemble(self, features: np.ndarray) -> Dict:
        """
        Make prediction using ensemble
        
        Args:
            features: Feature array
        
        Returns:
            Prediction result
        """
        return self.ensemble.predict(features)
    
    def add_xgboost_model(self, model_path: str, weight: float = 1.0):
        """
        Add XGBoost model to ensemble
        
        Args:
            model_path: Path to XGBoost model
            weight: Initial weight
        """
        from .xgboost_model import XGBoostModel
        
        model = XGBoostModel()
        model.load(model_path)
        self.ensemble.add_model('xgboost', model, weight)
    
    def add_bayesian_model(self, model_path: str, weight: float = 1.0):
        """
        Add Bayesian Neural Network model to ensemble
        
        Args:
            model_path: Path to Bayesian model
            weight: Initial weight
        """
        import torch
        from .bayesian_nn import BayesianTradingNetwork
        
        model = BayesianTradingNetwork(input_dim=50)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.ensemble.add_model('bayesian_nn', model, weight)
    
    def add_rl_model(self, model_path: str, weight: float = 1.0):
        """
        Add Reinforcement Learning model to ensemble
        
        Args:
            model_path: Path to RL model
            weight: Initial weight
        """
        import torch
        from .rl_agent import RLTrader
        
        model = RLTrader(state_dim=20, action_dim=3)
        model.actor.load_state_dict(torch.load(model_path))
        self.ensemble.add_model('rl_agent', model, weight)