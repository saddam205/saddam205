"""
drift_detection.py
Part of the app/monitoring module.
Detects data drift and concept drift in streaming market data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import jensenshannon
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Container for drift detection results"""
    has_drift: bool
    drift_score: float
    drift_type: str  # 'data', 'concept', 'feature'
    affected_features: List[str]
    severity: str  # 'low', 'medium', 'high'
    timestamp: datetime
    recommendation: str
    details: Dict[str, Any]


class DriftDetector:
    """
    Detects data drift and concept drift in streaming data.
    Uses statistical tests and distribution comparisons.
    """
    
    def __init__(self, window_size: int = 1000, 
                 reference_window: int = 5000,
                 drift_threshold: float = 0.05):
        """
        Initialize drift detector
        
        Args:
            window_size: Size of current window to check
            reference_window: Size of reference window (baseline)
            drift_threshold: Threshold for drift detection
        """
        self.window_size = window_size
        self.reference_window = reference_window
        self.drift_threshold = drift_threshold
        self.reference_data: Optional[np.ndarray] = None
        self.drift_history: List[DriftResult] = []
        
    def set_reference(self, data: np.ndarray):
        """
        Set reference data (baseline distribution)
        
        Args:
            data: Reference data array
        """
        self.reference_data = data
        logger.info(f"Reference data set with {len(data)} samples")
    
    def detect_data_drift(self, current_data: np.ndarray, 
                          feature_names: Optional[List[str]] = None) -> DriftResult:
        """
        Detect data drift using statistical tests
        
        Args:
            current_data: Current data window
            feature_names: Names of features (for reporting)
        
        Returns:
            DriftResult object
        """
        if self.reference_data is None:
            logger.warning("No reference data set")
            return DriftResult(
                has_drift=False,
                drift_score=0,
                drift_type='data',
                affected_features=[],
                severity='low',
                timestamp=datetime.now(),
                recommendation="Set reference data first",
                details={}
            )
        
        # Ensure same number of features
        if self.reference_data.shape[1] != current_data.shape[1]:
            logger.error("Feature dimensions mismatch")
            return DriftResult(
                has_drift=False,
                drift_score=0,
                drift_type='data',
                affected_features=[],
                severity='low',
                timestamp=datetime.now(),
                recommendation="Feature dimensions mismatch",
                details={}
            )
        
        # Check each feature for drift
        affected_features = []
        feature_scores = []
        
        for i in range(current_data.shape[1]):
            ref_feature = self.reference_data[:, i]
            curr_feature = current_data[:, i]
            
            # Perform Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref_feature, curr_feature)
            
            if p_value < self.drift_threshold:
                feature_name = feature_names[i] if feature_names else f"feature_{i}"
                affected_features.append(feature_name)
                feature_scores.append(ks_stat)
        
        # Calculate overall drift score
        drift_score = np.mean(feature_scores) if feature_scores else 0
        
        # Determine severity
        if len(affected_features) > current_data.shape[1] * 0.3:
            severity = 'high'
            recommendation = "Significant data drift detected. Consider retraining model."
        elif len(affected_features) > current_data.shape[1] * 0.1:
            severity = 'medium'
            recommendation = "Moderate data drift. Monitor closely."
        else:
            severity = 'low'
            recommendation = "Minor data drift detected. No action needed."
        
        result = DriftResult(
            has_drift=len(affected_features) > 0,
            drift_score=drift_score,
            drift_type='data',
            affected_features=affected_features,
            severity=severity,
            timestamp=datetime.now(),
            recommendation=recommendation,
            details={
                'affected_count': len(affected_features),
                'total_features': current_data.shape[1],
                'ks_statistics': dict(zip(affected_features, feature_scores))
            }
        )
        
        self.drift_history.append(result)
        
        if result.has_drift:
            logger.warning(f"Data drift detected: {len(affected_features)} features affected")
        
        return result
    
    def detect_concept_drift(self, predictions: np.ndarray, 
                            actuals: np.ndarray,
                            window_size: int = 100) -> DriftResult:
        """
        Detect concept drift using performance monitoring
        
        Args:
            predictions: Model predictions
            actuals: Actual outcomes
            window_size: Window size for performance monitoring
        
        Returns:
            DriftResult object
        """
        if len(predictions) < window_size * 2:
            return DriftResult(
                has_drift=False,
                drift_score=0,
                drift_type='concept',
                affected_features=[],
                severity='low',
                timestamp=datetime.now(),
                recommendation="Insufficient data",
                details={}
            )
        
        # Calculate performance over windows
        recent_accuracy = (predictions[-window_size:] == actuals[-window_size:]).mean()
        historical_accuracy = (predictions[:-window_size] == actuals[:-window_size]).mean()
        
        # Calculate performance drop
        performance_drop = historical_accuracy - recent_accuracy
        
        # Determine if concept drift occurred
        has_drift = performance_drop > 0.1  # 10% drop threshold
        
        severity = 'high' if performance_drop > 0.2 else 'medium' if performance_drop > 0.1 else 'low'
        
        result = DriftResult(
            has_drift=has_drift,
            drift_score=performance_drop,
            drift_type='concept',
            affected_features=['model_performance'],
            severity=severity,
            timestamp=datetime.now(),
            recommendation="Retrain model with recent data" if has_drift else "No action needed",
            details={
                'historical_accuracy': historical_accuracy,
                'recent_accuracy': recent_accuracy,
                'performance_drop': performance_drop,
                'window_size': window_size
            }
        )
        
        self.drift_history.append(result)
        
        if has_drift:
            logger.warning(f"Concept drift detected: Performance dropped {performance_drop:.2%}")
        
        return result
    
    def get_drift_report(self) -> Dict:
        """Generate drift detection report"""
        if not self.drift_history:
            return {'message': 'No drift detection history'}
        
        recent_drifts = [d for d in self.drift_history[-100:] if d.has_drift]
        
        return {
            'total_drifts': len(recent_drifts),
            'data_drifts': len([d for d in recent_drifts if d.drift_type == 'data']),
            'concept_drifts': len([d for d in recent_drifts if d.drift_type == 'concept']),
            'current_severity': recent_drifts[-1].severity if recent_drifts else 'none',
            'recommendations': list(set([d.recommendation for d in recent_drifts[-10:]])),
            'drift_timeline': [
                {
                    'timestamp': d.timestamp.isoformat(),
                    'type': d.drift_type,
                    'severity': d.severity,
                    'score': d.drift_score
                }
                for d in recent_drifts[-20:]
            ]
        }


class DataDriftMonitor:
    """
    Real-time data drift monitor for streaming features
    """
    
    def __init__(self, feature_distributions: Dict[str, Dict], 
                 update_interval: int = 60):
        """
        Initialize data drift monitor
        
        Args:
            feature_distributions: Reference distributions per feature
            update_interval: Update interval in seconds
        """
        self.feature_distributions = feature_distributions
        self.update_interval = update_interval
        self.current_stats: Dict[str, Dict] = {}
        self.drift_alerts = []
        
    def update_feature(self, feature_name: str, value: float):
        """
        Update feature with new value
        
        Args:
            feature_name: Name of feature
            value: New feature value
        """
        if feature_name not in self.current_stats:
            self.current_stats[feature_name] = {
                'values': [],
                'mean': 0,
                'std': 0,
                'count': 0
            }
        
        stats_dict = self.current_stats[feature_name]
        stats_dict['values'].append(value)
        stats_dict['count'] += 1
        stats_dict['mean'] = np.mean(stats_dict['values'][-1000:])
        stats_dict['std'] = np.std(stats_dict['values'][-1000:])
        
        # Check for drift periodically
        if stats_dict['count'] % self.update_interval == 0:
            self._check_drift(feature_name)
    
    def _check_drift(self, feature_name: str):
        """Check if feature has drifted"""
        if feature_name not in self.feature_distributions:
            return
        
        ref_dist = self.feature_distributions[feature_name]
        curr_stats = self.current_stats[feature_name]
        
        # Z-score based drift detection
        z_score = abs(curr_stats['mean'] - ref_dist['mean']) / (ref_dist['std'] + 1e-6)
        
        if z_score > 3:  # 3 sigma threshold
            self.drift_alerts.append({
                'feature': feature_name,
                'z_score': z_score,
                'timestamp': datetime.now(),
                'current_mean': curr_stats['mean'],
                'reference_mean': ref_dist['mean']
            })
            logger.warning(f"Drift detected in {feature_name}: Z-score={z_score:.2f}")
    
    def get_drift_status(self) -> Dict:
        """Get current drift status"""
        return {
            'features_monitored': len(self.current_stats),
            'recent_alerts': self.drift_alerts[-10:],
            'alert_count': len(self.drift_alerts),
            'healthy': len(self.drift_alerts) == 0
        }


class ConceptDriftDetector:
    """
    Advanced concept drift detection using ADWIN algorithm
    """
    
    def __init__(self, delta: float = 0.002, max_window: int = 1000):
        """
        Initialize ADWIN drift detector
        
        Args:
            delta: Confidence parameter (higher = more sensitive)
            max_window: Maximum window size
        """
        self.delta = delta
        self.max_window = max_window
        self.window = []
        self.drift_points = []
        
    def add_value(self, value: float):
        """
        Add new value to detector
        
        Args:
            value: New value (e.g., error or prediction)
        """
        self.window.append(value)
        
        # Trim window if too large
        if len(self.window) > self.max_window:
            self.window.pop(0)
        
        # Check for drift
        self._check_drift()
    
    def _check_drift(self):
        """Check for drift using ADWIN algorithm"""
        n = len(self.window)
        if n < 100:
            return
        
        # Check all possible split points
        for i in range(10, n - 10):
            window1 = self.window[:i]
            window2 = self.window[i:]
            
            # Calculate means
            mu1 = np.mean(window1)
            mu2 = np.mean(window2)
            
            # Calculate cut threshold
            m = 1 / (1/len(window1) + 1/len(window2))
            epsilon = np.sqrt(2 * m * np.log(2 / self.delta))
            
            # Check if difference exceeds threshold
            if abs(mu1 - mu2) > epsilon:
                self.drift_points.append({
                    'index': i,
                    'timestamp': datetime.now(),
                    'mean_before': mu1,
                    'mean_after': mu2,
                    'difference': abs(mu1 - mu2)
                })
                # Reset window after drift
                self.window = self.window[i:]
                logger.info(f"Concept drift detected at index {i}")
                break
    
    def has_drift(self) -> bool:
        """Check if drift has been detected"""
        return len(self.drift_points) > 0
    
    def get_last_drift(self) -> Optional[Dict]:
        """Get last drift point"""
        return self.drift_points[-1] if self.drift_points else None