"""
walk_forward.py
Part of the app/backtesting module.
Walk-forward validation to prevent overfitting.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """
    Professional walk-forward validation for trading systems
    """
    
    def __init__(self, model_class, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        self.model_class = model_class
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.results = []
        self.aggregate_metrics = {}
        
    def validate(self, data, n_splits=5, retrain_frequency='quarterly'):
        """
        Run walk-forward validation
        
        Args:
            data: DataFrame with datetime index and features
            n_splits: Number of forward windows
            retrain_frequency: How often to retrain ('monthly', 'quarterly', 'yearly')
        """
        logger.info("="*60)
        logger.info("Starting Walk-Forward Validation")
        logger.info("="*60)
        logger.info(f"Total samples: {len(data)}")
        logger.info(f"Number of windows: {n_splits}")
        
        # Calculate window sizes
        total_samples = len(data)
        train_size = int(total_samples * self.train_ratio)
        val_size = int(total_samples * self.val_ratio)
        test_size = int(total_samples * self.test_ratio)
        
        results = []
        
        for window in range(n_splits):
            logger.info(f"\n📊 Window {window + 1}/{n_splits}")
            logger.info("-" * 40)
            
            # Define time windows (no look-ahead!)
            train_end = train_size + (window * test_size)
            val_end = train_end + val_size
            test_end = val_end + test_size
            
            if test_end > len(data):
                logger.warning("Reached end of data. Stopping.")
                break
            
            # Split data chronologically
            train_data = data.iloc[:train_end]
            val_data = data.iloc[train_end:val_end]
            test_data = data.iloc[val_end:test_end]
            
            logger.info(f"Train: {train_data.index[0]} to {train_data.index[-1]}")
            logger.info(f"Val: {val_data.index[0]} to {val_data.index[-1]}")
            logger.info(f"Test: {test_data.index[0]} to {test_data.index[-1]}")
            
            # Train model
            model = self.model_class()
            model.fit(train_data, val_data)
            
            # Test on out-of-sample data
            test_results = model.evaluate(test_data)
            
            # Store results
            window_result = {
                'window': window + 1,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'accuracy': test_results.get('accuracy', 0),
                'sharpe': test_results.get('sharpe_ratio', 0),
                'max_drawdown': test_results.get('max_drawdown', 0),
                'win_rate': test_results.get('win_rate', 0),
                'profit_factor': test_results.get('profit_factor', 0),
                'trades': test_results.get('total_trades', 0)
            }
            
            results.append(window_result)
            
            logger.info(f"Results:")
            logger.info(f"  Accuracy: {window_result['accuracy']:.2%}")
            logger.info(f"  Sharpe: {window_result['sharpe']:.2f}")
            logger.info(f"  Max DD: {window_result['max_drawdown']:.2%}")
        
        # Aggregate results
        self.results = results
        self._calculate_aggregate_metrics()
        
        return results
    
    def _calculate_aggregate_metrics(self):
        """Calculate aggregate performance metrics"""
        if not self.results:
            return
        
        accuracies = [r['accuracy'] for r in self.results]
        sharpes = [r['sharpe'] for r in self.results]
        drawdowns = [r['max_drawdown'] for r in self.results]
        win_rates = [r['win_rate'] for r in self.results]
        
        self.aggregate_metrics = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'accuracy_stability': 1 - (np.std(accuracies) / (np.mean(accuracies) + 1e-6)),
            'mean_sharpe': np.mean(sharpes),
            'mean_max_drawdown': np.mean(drawdowns),
            'max_drawdown_peak': np.max(drawdowns),
            'mean_win_rate': np.mean(win_rates),
            'robustness_score': self._calculate_robustness_score()
        }
        
        logger.info("\n" + "="*60)
        logger.info("📊 AGGREGATE WALK-FORWARD RESULTS")
        logger.info("="*60)
        logger.info(f"Mean Accuracy: {self.aggregate_metrics['mean_accuracy']:.2%} ± {self.aggregate_metrics['std_accuracy']:.2%}")
        logger.info(f"Accuracy Range: {self.aggregate_metrics['min_accuracy']:.2%} - {self.aggregate_metrics['max_accuracy']:.2%}")
        logger.info(f"Accuracy Stability: {self.aggregate_metrics['accuracy_stability']:.2%}")
        logger.info(f"Mean Sharpe Ratio: {self.aggregate_metrics['mean_sharpe']:.2f}")
        logger.info(f"Mean Max Drawdown: {self.aggregate_metrics['mean_max_drawdown']:.2%}")
        logger.info(f"Worst Drawdown: {self.aggregate_metrics['max_drawdown_peak']:.2%}")
        logger.info(f"Mean Win Rate: {self.aggregate_metrics['mean_win_rate']:.2%}")
        logger.info(f"\nRobustness Score: {self.aggregate_metrics['robustness_score']:.2f}/1.00")
        
        # Final verdict
        if self.aggregate_metrics['robustness_score'] > 0.7:
            logger.info("\n✅ SYSTEM IS ROBUST - Passes walk-forward validation")
        elif self.aggregate_metrics['robustness_score'] > 0.5:
            logger.info("\n⚠️ SYSTEM IS MODERATELY ROBUST - Use with caution")
        else:
            logger.info("\n❌ SYSTEM IS OVERFITTED - Needs improvement")
    
    def _calculate_robustness_score(self):
        """Calculate overall robustness score"""
        # Read from already-populated aggregate_metrics keys
        accuracy_stability = self.aggregate_metrics.get('accuracy_stability', 0)
        sharpe_score = min(self.aggregate_metrics.get('mean_sharpe', 0) / 2, 1)
        drawdown_score = 1 - min(
            self.aggregate_metrics.get('mean_max_drawdown', 0) / 0.2, 1
        )
        robustness = (accuracy_stability * 0.4 +
                      sharpe_score * 0.4 +
                      drawdown_score * 0.2)
        return robustness
    
    def get_best_window(self):
        """Get the best performing window"""
        if not self.results:
            return None
        
        best_idx = np.argmax([r['sharpe'] for r in self.results])
        return self.results[best_idx]
    
    def get_worst_window(self):
        """Get the worst performing window"""
        if not self.results:
            return None
        
        worst_idx = np.argmin([r['sharpe'] for r in self.results])
        return self.results[worst_idx]