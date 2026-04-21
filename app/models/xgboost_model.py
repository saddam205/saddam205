"""
xgboost_model.py
Part of the app/models module.
XGBoost implementation for trading signal prediction.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost model for trading signal prediction with time series validation
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize XGBoost model
        
        Args:
            params: XGBoost parameters (defaults optimized for trading)
        """
        self.params = params or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = None
        self.feature_importance = None
        self.training_history = []
        self.best_score = float('inf')
        
    def prepare_features(self, data: pd.DataFrame, 
                        feature_columns: List[str],
                        target_column: str = 'signal',
                        lookback: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features with time series alignment
        
        Args:
            data: Input DataFrame
            feature_columns: List of feature column names
            target_column: Target column name
            lookback: Lookback period for feature engineering
        
        Returns:
            Tuple of (features, targets)
        """
        # Create lagged features
        X = data[feature_columns].copy()
        
        for col in feature_columns:
            for lag in [1, 2, 3, 5, 10]:
                X[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        # Add rolling statistics
        for col in feature_columns:
            X[f'{col}_rolling_mean_5'] = data[col].rolling(5).mean()
            X[f'{col}_rolling_std_5'] = data[col].rolling(5).std()
            X[f'{col}_rolling_mean_10'] = data[col].rolling(10).mean()
            X[f'{col}_rolling_std_10'] = data[col].rolling(10).std()
        
        # Drop NaN values
        X = X.dropna()
        y = data.loc[X.index, target_column]
        
        # Encode target if needed
        if y.dtype == 'object':
            unique_labels = y.unique()
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y = y.map(label_map)
        
        return X.values, y.values
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2,
              early_stopping_rounds: int = 50) -> Dict:
        """
        Train XGBoost model with time series validation
        
        Args:
            X: Feature matrix
            y: Target labels
            validation_split: Proportion of data for validation
            early_stopping_rounds: Early stopping patience
        
        Returns:
            Training metrics
        """
        # Time series split (no look-ahead)
        split_idx = int(len(X) * (1 - validation_split))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train with early stopping
        evals = [(dtrain, 'train'), (dval, 'eval')]
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50
        )
        
        # Calculate feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')
        
        # Evaluate on validation set
        val_pred = self.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        metrics = {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'validation_accuracy': val_accuracy,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'num_features': X.shape[1]
        }
        
        self.training_history.append(metrics)
        self.best_score = self.model.best_score
        
        logger.info(f"Model trained: {metrics}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict trading signals
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        dtest = xgb.DMatrix(X)
        predictions = self.model.predict(dtest)
        
        # For multi-class, get argmax
        if len(predictions.shape) > 1:
            return np.argmax(predictions, axis=1)
        else:
            # For binary classification, threshold at 0.5
            return (predictions > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix
        
        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def predict_with_confidence(self, X: np.ndarray) -> List[Dict]:
        """
        Predict with confidence scores
        
        Args:
            X: Feature matrix
        
        Returns:
            List of predictions with confidence
        """
        probabilities = self.predict_proba(X)
        
        results = []
        for prob in probabilities:
            if len(prob.shape) == 0:  # Binary case
                prob_array = np.array([1 - prob, prob])
            else:
                prob_array = prob
            
            signal_idx = np.argmax(prob_array)
            confidence = prob_array[signal_idx]
            
            # Map index to signal
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            signal = signal_map.get(signal_idx, 'HOLD')
            
            results.append({
                'signal': signal,
                'confidence': float(confidence),
                'probabilities': prob_array.tolist()
            })
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get feature importance ranking
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            List of (feature_name, importance)
        """
        if self.feature_importance is None:
            return []
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_features[:top_n]
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       n_splits: int = 5) -> Dict:
        """
        Perform time series cross-validation
        
        Args:
            X: Feature matrix
            y: Target labels
            n_splits: Number of CV splits
        
        Returns:
            Cross-validation metrics
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=100,
                evals=[(dval, 'eval')],
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            # Predict
            pred = self.predict(X_val)
            
            # Calculate metrics
            cv_scores['accuracy'].append(accuracy_score(y_val, pred))
            cv_scores['precision'].append(precision_score(y_val, pred, average='weighted', zero_division=0))
            cv_scores['recall'].append(recall_score(y_val, pred, average='weighted', zero_division=0))
            cv_scores['f1'].append(f1_score(y_val, pred, average='weighted', zero_division=0))
            
            logger.info(f"Fold {fold + 1}: Accuracy={cv_scores['accuracy'][-1]:.4f}")
        
        # Aggregate results
        results = {
            'mean_accuracy': np.mean(cv_scores['accuracy']),
            'std_accuracy': np.std(cv_scores['accuracy']),
            'mean_precision': np.mean(cv_scores['precision']),
            'mean_recall': np.mean(cv_scores['recall']),
            'mean_f1': np.mean(cv_scores['f1']),
            'fold_results': cv_scores
        }
        
        logger.info(f"CV Results: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")
        
        return results
    
    def save(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save model
        """
        if self.model:
            self.model.save_model(filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from disk
        
        Args:
            filepath: Path to load model from
        """
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class XGBoostPredictor:
    """
    High-level XGBoost predictor for live trading
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to pre-trained model
        """
        self.model = XGBoostModel()
        self.feature_columns = []
        self.scaler = None
        
        if model_path:
            self.load(model_path)
    
    def prepare_live_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for live prediction
        
        Args:
            data: Latest market data
        
        Returns:
            Feature array for prediction
        """
        if not self.feature_columns:
            raise ValueError("Feature columns not set. Load model first.")
        
        # Extract latest features
        latest = data[self.feature_columns].iloc[-1:].copy()
        
        # Add lagged features
        for col in self.feature_columns:
            for lag in [1, 2, 3, 5, 10]:
                if len(data) > lag:
                    latest[f'{col}_lag_{lag}'] = data[col].iloc[-lag]
                else:
                    latest[f'{col}_lag_{lag}'] = 0
        
        # Add rolling statistics
        for col in self.feature_columns:
            if len(data) >= 5:
                latest[f'{col}_rolling_mean_5'] = data[col].tail(5).mean()
                latest[f'{col}_rolling_std_5'] = data[col].tail(5).std()
            else:
                latest[f'{col}_rolling_mean_5'] = data[col].mean()
                latest[f'{col}_rolling_std_5'] = data[col].std()
        
        return latest.values
    
    def predict_live(self, data: pd.DataFrame) -> Dict:
        """
        Make live prediction
        
        Args:
            data: Latest market data
        
        Returns:
            Prediction result
        """
        features = self.prepare_live_features(data)
        result = self.model.predict_with_confidence(features)[0]
        
        return result
    
    def save(self, filepath: str, metadata: Dict = None):
        """
        Save model and metadata
        
        Args:
            filepath: Path to save model
            metadata: Additional metadata to save
        """
        self.model.save(filepath)
        
        if metadata:
            metadata_path = filepath.replace('.model', '_metadata.pkl')
            joblib.dump({
                'feature_columns': self.feature_columns,
                'metadata': metadata
            }, metadata_path)
    
    def load(self, filepath: str):
        """
        Load model and metadata
        
        Args:
            filepath: Path to load model from
        """
        self.model.load(filepath)
        
        metadata_path = filepath.replace('.model', '_metadata.pkl')
        try:
            metadata = joblib.load(metadata_path)
            self.feature_columns = metadata.get('feature_columns', [])
        except:
            logger.warning("No metadata found for model")