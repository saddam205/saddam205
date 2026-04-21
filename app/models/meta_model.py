"""
meta_model.py
Part of the app/models module.
Meta-model that decides WHEN to trade (market regime and condition filtering).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MetaModel:
    """
    Meta-model that learns when NOT to trade.
    Acts as a gatekeeper for the trading system.
    """
    
    def __init__(self):
        """Initialize meta-model"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
        self.threshold = 0.6  # Minimum confidence to trade
        
    def prepare_meta_features(self, market_data: pd.DataFrame, 
                              predictions: Dict) -> np.ndarray:
        """
        Prepare features for meta-model
        
        Args:
            market_data: Current market data
            predictions: Model predictions from ensemble
        
        Returns:
            Feature array for meta-model
        """
        features = []
        
        # Market condition features
        if len(market_data) >= 20:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.tail(20).std()
            features.append(volatility)
            
            # Trend strength
            sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
            sma_50 = market_data['close'].rolling(50).mean().iloc[-1]
            trend_strength = abs(sma_20 / sma_50 - 1)
            features.append(trend_strength)
            
            # Volume anomaly
            avg_volume = market_data['volume'].tail(20).mean()
            current_volume = market_data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            features.append(min(volume_ratio, 5))
        else:
            features.extend([0.02, 0.01, 1.0])
        
        # Prediction features
        features.append(predictions.get('confidence', 0.5))
        features.append(predictions.get('ensemble_size', 1) / 5)  # Normalized
        
        # Signal scores
        signal_scores = predictions.get('signal_scores', {})
        features.append(signal_scores.get('BUY', 0))
        features.append(signal_scores.get('SELL', 0))
        features.append(signal_scores.get('HOLD', 0))
        
        # Model agreement
        model_predictions = predictions.get('model_predictions', [])
        if model_predictions:
            signals = [p['prediction'] for p in model_predictions]
            agreement = signals.count(signals[0]) / len(signals) if signals else 0
            features.append(agreement)
        else:
            features.append(0)
        
        # Time features
        current_hour = datetime.now().hour
        features.append(current_hour / 24)
        features.append(1 if current_hour < 9 or current_hour > 16 else 0)  # After hours
        
        return np.array(features).reshape(1, -1)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train meta-model
        
        Args:
            X: Feature matrix (market conditions + predictions)
            y: Target (1 = should trade, 0 = should not trade)
        
        Returns:
            Training metrics
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            [f'feature_{i}' for i in range(X.shape[1])],
            self.model.feature_importances_
        ))
        
        # Calculate training accuracy
        train_pred = self.model.predict(X_scaled)
        accuracy = (train_pred == y).mean()
        
        logger.info(f"Meta-model trained with accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'feature_importance': self.feature_importance,
            'n_features': X.shape[1],
            'n_samples': len(y)
        }
    
    def should_trade(self, predictions: Dict, market_data: pd.DataFrame) -> bool:
        """
        Decide whether to trade based on current conditions
        
        Args:
            predictions: Model predictions from ensemble
            market_data: Current market data
        
        Returns:
            True if should trade, False otherwise
        """
        if not self.is_trained:
            # Default logic if not trained
            return predictions.get('confidence', 0) > 0.7
        
        # Prepare features
        features = self.prepare_meta_features(market_data, predictions)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        proba = self.model.predict_proba(features_scaled)[0]
        trade_probability = proba[1] if len(proba) > 1 else proba[0]
        
        should_trade = trade_probability > self.threshold
        
        logger.debug(f"Meta-model decision: trade={should_trade}, probability={trade_probability:.3f}")
        
        return should_trade
    
    def get_trade_probability(self, predictions: Dict, 
                              market_data: pd.DataFrame) -> float:
        """
        Get probability that trading is favorable
        
        Args:
            predictions: Model predictions
            market_data: Current market data
        
        Returns:
            Trade probability (0-1)
        """
        if not self.is_trained:
            return predictions.get('confidence', 0.5)
        
        features = self.prepare_meta_features(market_data, predictions)
        features_scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(features_scaled)[0]
        
        return proba[1] if len(proba) > 1 else proba[0]
    
    def set_threshold(self, threshold: float):
        """
        Set trade decision threshold
        
        Args:
            threshold: Threshold value (0-1)
        """
        self.threshold = max(0, min(1, threshold))
        logger.info(f"Meta-model threshold set to {threshold}")
    
    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """
        Get feature importance ranking
        
        Returns:
            List of (feature_name, importance)
        """
        if not self.feature_importance:
            return []
        
        return sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    def save(self, filepath: str):
        """
        Save meta-model to disk
        
        Args:
            filepath: Path to save model
        """
        import joblib
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Meta-model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load meta-model from disk
        
        Args:
            filepath: Path to load model from
        """
        import joblib
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.threshold = model_data.get('threshold', 0.6)
        self.feature_importance = model_data.get('feature_importance')
        self.is_trained = model_data.get('is_trained', True)
        
        logger.info(f"Meta-model loaded from {filepath}")


class WhenNotToTrade:
    """
    Rule-based filter for identifying unfavorable trading conditions
    """
    
    def __init__(self):
        """Initialize rule-based filter"""
        self.rules = []
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup default rules"""
        self.rules = [
            ('high_volatility', self._check_volatility),
            ('low_liquidity', self._check_liquidity),
            ('major_news', self._check_major_news),
            ('market_hours', self._check_market_hours),
            ('extreme_sentiment', self._check_extreme_sentiment)
        ]
    
    def _check_volatility(self, market_data: pd.DataFrame) -> Tuple[bool, str]:
        """Check if volatility is too high"""
        if len(market_data) >= 20:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.tail(20).std() * np.sqrt(252)
            
            if volatility > 0.05:  # 50% annualized volatility threshold
                return True, f"High volatility: {volatility:.2%}"
        
        return False, ""
    
    def _check_liquidity(self, market_data: pd.DataFrame) -> Tuple[bool, str]:
        """Check if liquidity is sufficient"""
        if len(market_data) >= 20:
            avg_volume = market_data['volume'].tail(20).mean()
            current_volume = market_data['volume'].iloc[-1]
            
            if current_volume < avg_volume * 0.5:
                return True, f"Low liquidity: volume {current_volume:.0f} vs avg {avg_volume:.0f}"
        
        return False, ""
    
    def _check_major_news(self, market_data: pd.DataFrame) -> Tuple[bool, str]:
        """Check for major news events (simplified)"""
        # In production, this would check news APIs
        current_hour = datetime.now().hour
        if 8 <= current_hour <= 10:  # Major news typically released 8-10 AM
            return True, "Major news expected during market hours"
        
        return False, ""
    
    def _check_market_hours(self, market_data: pd.DataFrame) -> Tuple[bool, str]:
        """Check if market is open"""
        current_hour = datetime.now().hour
        if current_hour < 9 or current_hour > 16:
            return True, "Market closed"
        
        return False, ""
    
    def _check_extreme_sentiment(self, market_data: pd.DataFrame) -> Tuple[bool, str]:
        """Check for extreme market sentiment"""
        if len(market_data) >= 20:
            # Use RSI as sentiment proxy
            delta = market_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            if not rsi.empty and rsi.iloc[-1] > 85:
                return True, f"Extreme overbought: RSI={rsi.iloc[-1]:.1f}"
            elif not rsi.empty and rsi.iloc[-1] < 15:
                return True, f"Extreme oversold: RSI={rsi.iloc[-1]:.1f}"
        
        return False, ""
    
    def evaluate(self, market_data: pd.DataFrame) -> Dict:
        """
        Evaluate all rules
        
        Args:
            market_data: Current market data
        
        Returns:
            Evaluation results
        """
        results = {
            'should_trade': True,
            'blocked_rules': [],
            'warnings': []
        }
        
        for rule_name, rule_func in self.rules:
            try:
                blocked, reason = rule_func(market_data)
                if blocked:
                    results['should_trade'] = False
                    results['blocked_rules'].append({
                        'rule': rule_name,
                        'reason': reason
                    })
                elif reason:
                    results['warnings'].append({
                        'rule': rule_name,
                        'reason': reason
                    })
            except Exception as e:
                logger.error(f"Rule {rule_name} failed: {e}")
        
        return results