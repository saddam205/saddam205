#!/usr/bin/env python3
"""
train_advanced_model.py
Enhanced training script for advanced AI models with hyperparameter optimization.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
import argparse
import json
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')


class AdvancedModelTrainer:
    """Advanced model training with hyperparameter optimization"""
    
    def __init__(self, symbol: str = "BTC-USD", output_dir: str = "data/models/"):
        """
        Initialize trainer
        
        Args:
            symbol: Trading symbol
            output_dir: Output directory for models
        """
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_data(self, interval: str = "15m", period: str = "90d") -> pd.DataFrame:
        """Fetch historical data"""
        print(f"\n📥 Downloading {interval} data for {self.symbol}...")
        data = yf.download(self.symbol, interval=interval, period=period)
        print(f"✅ Data shape: {data.shape}")
        return data
    
    def calculate_indicators(self, df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = df.copy()
        
        # Moving Averages
        for period in [10, 20, 50, 200]:
            df[f'{prefix}SMA{period}'] = df['Close'].rolling(period).mean()
        
        # EMAs
        df[f'{prefix}EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df[f'{prefix}EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df[f'{prefix}MACD'] = df[f'{prefix}EMA12'] - df[f'{prefix}EMA26']
        df[f'{prefix}MACD_Signal'] = df[f'{prefix}MACD'].ewm(span=9, adjust=False).mean()
        df[f'{prefix}MACD_Hist'] = df[f'{prefix}MACD'] - df[f'{prefix}MACD_Signal']
        
        # RSI (multiple periods)
        for period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'{prefix}RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df[f'{prefix}BB_Mid'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df[f'{prefix}BB_Upper'] = df[f'{prefix}BB_Mid'] + (bb_std * 2)
        df[f'{prefix}BB_Lower'] = df[f'{prefix}BB_Mid'] - (bb_std * 2)
        df[f'{prefix}BB_Width'] = (df[f'{prefix}BB_Upper'] - df[f'{prefix}BB_Lower']) / df[f'{prefix}BB_Mid']
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'{prefix}ATR'] = tr.rolling(14).mean()
        df[f'{prefix}ATR_Pct'] = df[f'{prefix}ATR'] / df['Close']
        
        # Volume
        df[f'{prefix}Volume_SMA'] = df['Volume'].rolling(20).mean()
        df[f'{prefix}Volume_Ratio'] = df['Volume'] / df[f'{prefix}Volume_SMA']
        
        # Price changes
        for period in [1, 3, 5, 10]:
            df[f'{prefix}Price_Change_{period}'] = df['Close'].pct_change(period)
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> tuple:
        """Create feature set and target"""
        # Normalize features
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        for col in feature_cols:
            df[f'{col}_norm'] = (df[col] - df[col].rolling(100).mean()) / (df[col].rolling(100).std() + 1e-8)
        
        # Create target (predict 3 periods ahead)
        df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
        
        # Clean data
        df = df.dropna()
        
        # Get normalized feature columns
        norm_features = [f'{col}_norm' for col in feature_cols if f'{col}_norm' in df.columns]
        
        X = df[norm_features]
        y = df['Target']
        
        print(f"✅ Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        return X, y, norm_features
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, optimize: bool = False) -> XGBClassifier:
        """Train XGBoost model with optional hyperparameter optimization"""
        
        if optimize:
            print("\n🔍 Running hyperparameter optimization...")
            
            param_grid = {
                'max_depth': [5, 7, 9],
                'learning_rate': [0.01, 0.02, 0.05],
                'n_estimators': [300, 500, 800],
                'subsample': [0.8, 0.85, 0.9],
                'colsample_bytree': [0.8, 0.85, 0.9]
            }
            
            grid_search = GridSearchCV(
                XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            print(f"✅ Best parameters: {best_params}")
            
            model = XGBClassifier(
                **best_params,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        else:
            model = XGBClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.02,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, feature_names: list):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 Model Evaluation:")
        print("=" * 50)
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives: {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives: {tp}")
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n🎯 Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Feature Importance
        importances = pd.Series(model.feature_importances_, index=feature_names)
        print("\n🔑 Top 10 Most Important Features:")
        print(importances.sort_values(ascending=False).head(10))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }
    
    def save_model(self, model, feature_names: list, metrics: dict, version: str = None):
        """Save model and metadata"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = self.output_dir / f"xgboost_model_{version}.pkl"
        features_path = self.output_dir / f"features_{version}.txt"
        metadata_path = self.output_dir / f"metadata_{version}.json"
        
        # Save model
        joblib.dump(model, model_path)
        print(f"\n💾 Model saved to {model_path}")
        
        # Save features
        with open(features_path, 'w') as f:
            for feature in feature_names:
                f.write(f"{feature}\n")
        print(f"✅ Features saved to {features_path}")
        
        # Save metadata
        metadata = {
            'version': version,
            'symbol': self.symbol,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics,
            'num_features': len(feature_names),
            'model_type': 'xgboost'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved to {metadata_path}")
        
        # Create symlink to latest
        latest_model = self.output_dir / "xgboost_model_latest.pkl"
        latest_features = self.output_dir / "features_latest.txt"
        
        if latest_model.exists():
            latest_model.unlink()
        if latest_features.exists():
            latest_features.unlink()
        
        latest_model.symlink_to(model_path.name)
        latest_features.symlink_to(features_path.name)
        
        print(f"✅ Latest model symlink created")
        
        return version
    
    def run(self, optimize: bool = False, train_split: float = 0.7):
        """Run complete training pipeline"""
        print("=" * 60)
        print("🤖 ADVANCED MODEL TRAINING PIPELINE")
        print("=" * 60)
        
        # Fetch data
        df = self.fetch_data()
        
        # Calculate indicators
        print("\n📊 Calculating indicators...")
        df = self.calculate_indicators(df)
        
        # Create features
        print("\n🛠️ Creating features...")
        X, y, feature_names = self.create_features(df)
        
        # Train/test split
        split_idx = int(len(X) * train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Further split train into train/val
        val_idx = int(len(X_train) * 0.8)
        X_train_sub, X_val = X_train[:val_idx], X_train[val_idx:]
        y_train_sub, y_val = y_train[:val_idx], y_train[val_idx:]
        
        print(f"\n📊 Data Split:")
        print(f"  Training: {len(X_train_sub)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Testing: {len(X_test)} samples")
        
        # Train model
        print("\n🚀 Training XGBoost model...")
        model = self.train_xgboost(X_train_sub, y_train_sub, X_val, y_val, optimize=optimize)
        
        # Evaluate
        metrics = self.evaluate_model(model, X_test, y_test, feature_names)
        
        # Save model
        version = self.save_model(model, feature_names, metrics)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Model Version: {version}")
        print(f"Test Accuracy: {metrics['accuracy']:.2%}")
        
        if metrics['accuracy'] >= 0.62:
            print("\n✅ TARGET ACHIEVED! Model accuracy is 62%+")
        else:
            print(f"\n⚠️ Current accuracy: {metrics['accuracy']:.2%}")
        
        return model, metrics


def main():
    parser = argparse.ArgumentParser(description='Train advanced AI trading model')
    parser.add_argument('--symbol', type=str, default='BTC-USD', help='Trading symbol')
    parser.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--output-dir', type=str, default='data/models/', help='Output directory')
    
    args = parser.parse_args()
    
    trainer = AdvancedModelTrainer(symbol=args.symbol, output_dir=args.output_dir)
    trainer.run(optimize=args.optimize)


if __name__ == "__main__":
    main()