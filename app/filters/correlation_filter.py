"""
correlation_filter.py
Part of the app/filters module.
Correlation-based filtering for portfolio diversification and risk management.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AssetCorrelation:
    """Asset correlation data"""
    asset1: str
    asset2: str
    correlation: float
    p_value: float
    sample_size: int
    is_significant: bool
    relationship: str  # 'positive', 'negative', 'uncorrelated'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'asset1': self.asset1,
            'asset2': self.asset2,
            'correlation': self.correlation,
            'p_value': self.p_value,
            'sample_size': self.sample_size,
            'is_significant': self.is_significant,
            'relationship': self.relationship
        }


class CorrelationFilter:
    """
    Correlation-based filter for portfolio management.
    Ensures diversification and manages correlation risk.
    """
    
    def __init__(self, max_correlation: float = 0.7, 
                 min_correlation_threshold: float = 0.3,
                 significance_level: float = 0.05):
        """
        Initialize correlation filter
        
        Args:
            max_correlation: Maximum allowed correlation between assets
            min_correlation_threshold: Minimum correlation to consider significant
            significance_level: Statistical significance level
        """
        self.max_correlation = max_correlation
        self.min_correlation_threshold = min_correlation_threshold
        self.significance_level = significance_level
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.asset_returns: Dict[str, pd.Series] = {}
        
    def update_correlations(self, price_data: Dict[str, pd.DataFrame],
                           lookback_days: int = 60) -> pd.DataFrame:
        """
        Update correlation matrix with latest data
        
        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            lookback_days: Number of days to look back
        
        Returns:
            Correlation matrix
        """
        # Calculate returns for each asset
        returns_dict = {}
        
        for symbol, data in price_data.items():
            if 'close' in data.columns and len(data) >= lookback_days:
                returns = data['close'].pct_change().dropna().tail(lookback_days)
                returns_dict[symbol] = returns
                self.asset_returns[symbol] = returns
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate correlation matrix
        self.correlation_matrix = returns_df.corr()
        
        logger.info(f"Updated correlation matrix for {len(returns_dict)} assets")
        
        return self.correlation_matrix
    
    def is_diversified(self, current_positions: List[str], 
                      new_asset: str) -> Tuple[bool, List[AssetCorrelation]]:
        """
        Check if adding a new asset maintains diversification
        
        Args:
            current_positions: List of current asset symbols
            new_asset: New asset to evaluate
        
        Returns:
            Tuple of (is_diversified, list_of_correlations)
        """
        if self.correlation_matrix is None:
            logger.warning("Correlation matrix not initialized")
            return True, []
        
        if new_asset not in self.correlation_matrix.index:
            logger.warning(f"Asset {new_asset} not in correlation matrix")
            return True, []
        
        correlations = []
        
        for asset in current_positions:
            if asset in self.correlation_matrix.columns:
                corr_value = self.correlation_matrix.loc[new_asset, asset]
                
                # Calculate significance (simplified)
                p_value = self._estimate_p_value(corr_value, len(self.asset_returns.get(asset, [])))
                is_significant = p_value < self.significance_level
                
                if abs(corr_value) > self.min_correlation_threshold:
                    relationship = 'positive' if corr_value > 0 else 'negative'
                else:
                    relationship = 'uncorrelated'
                
                asset_corr = AssetCorrelation(
                    asset1=new_asset,
                    asset2=asset,
                    correlation=corr_value,
                    p_value=p_value,
                    sample_size=len(self.asset_returns.get(asset, [])),
                    is_significant=is_significant,
                    relationship=relationship
                )
                correlations.append(asset_corr)
        
        # Check if any correlation exceeds max
        high_correlations = [
            c for c in correlations 
            if abs(c.correlation) > self.max_correlation and c.is_significant
        ]
        
        is_diversified = len(high_correlations) == 0
        
        if not is_diversified:
            logger.debug(f"Asset {new_asset} has high correlation with {len(high_correlations)} existing positions")
        
        return is_diversified, correlations
    
    def filter_by_correlation(self, signals: List[Dict], 
                             current_positions: List[str]) -> List[Dict]:
        """
        Filter trading signals based on correlation with existing positions
        
        Args:
            signals: List of trading signals
            current_positions: List of current asset symbols
        
        Returns:
            Filtered signals list
        """
        if not current_positions:
            return signals
        
        filtered_signals = []
        
        for signal in signals:
            symbol = signal.get('symbol')
            if not symbol:
                continue
            
            is_diversified, correlations = self.is_diversified(current_positions, symbol)
            
            if is_diversified:
                filtered_signals.append(signal)
                signal['correlation_check'] = {
                    'passed': True,
                    'correlations': [c.to_dict() for c in correlations]
                }
            else:
                signal['correlation_check'] = {
                    'passed': False,
                    'reason': f"High correlation with existing positions",
                    'correlations': [c.to_dict() for c in correlations if abs(c.correlation) > self.max_correlation]
                }
                logger.debug(f"Signal for {symbol} rejected due to correlation")
        
        return filtered_signals
    
    def get_correlation_risk(self, positions: List[str]) -> Dict:
        """
        Calculate correlation risk for a portfolio
        
        Args:
            positions: List of asset symbols
        
        Returns:
            Correlation risk metrics
        """
        if self.correlation_matrix is None or len(positions) < 2:
            return {
                'average_correlation': 0,
                'max_correlation': 0,
                'diversification_ratio': 1,
                'risk_score': 0
            }
        
        correlations = []
        
        for i, asset1 in enumerate(positions):
            for asset2 in positions[i+1:]:
                if asset1 in self.correlation_matrix.index and asset2 in self.correlation_matrix.columns:
                    corr = self.correlation_matrix.loc[asset1, asset2]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        if not correlations:
            return {
                'average_correlation': 0,
                'max_correlation': 0,
                'diversification_ratio': 1,
                'risk_score': 0
            }
        
        avg_corr = np.mean(correlations)
        max_corr = np.max(correlations)
        
        # Diversification ratio (lower is better)
        diversification_ratio = avg_corr
        
        # Risk score (0-1, higher means more correlation risk)
        risk_score = min(1, avg_corr / self.max_correlation)
        
        return {
            'average_correlation': avg_corr,
            'max_correlation': max_corr,
            'diversification_ratio': diversification_ratio,
            'risk_score': risk_score,
            'num_pairs': len(correlations)
        }
    
    def find_uncorrelated_assets(self, target_asset: str, 
                                 candidate_assets: List[str],
                                 max_correlation: float = None) -> List[str]:
        """
        Find assets with low correlation to target
        
        Args:
            target_asset: Target asset symbol
            candidate_assets: List of candidate assets
            max_correlation: Maximum allowed correlation (defaults to instance value)
        
        Returns:
            List of uncorrelated assets
        """
        if self.correlation_matrix is None:
            logger.warning("Correlation matrix not initialized")
            return candidate_assets
        
        if target_asset not in self.correlation_matrix.index:
            logger.warning(f"Target asset {target_asset} not in correlation matrix")
            return candidate_assets
        
        max_corr = max_correlation or self.max_correlation
        
        uncorrelated = []
        
        for asset in candidate_assets:
            if asset in self.correlation_matrix.columns:
                corr = abs(self.correlation_matrix.loc[target_asset, asset])
                if not np.isnan(corr) and corr <= max_corr:
                    uncorrelated.append(asset)
        
        # Sort by correlation (lowest first)
        uncorrelated.sort(key=lambda x: abs(self.correlation_matrix.loc[target_asset, x]))
        
        return uncorrelated
    
    def get_hedge_ratio(self, asset1: str, asset2: str) -> float:
        """
        Calculate optimal hedge ratio between two assets
        
        Args:
            asset1: First asset
            asset2: Second asset (hedge)
        
        Returns:
            Hedge ratio (amount of asset2 to hedge 1 unit of asset1)
        """
        if self.correlation_matrix is None:
            return 1.0
        
        if asset1 not in self.asset_returns or asset2 not in self.asset_returns:
            return 1.0
        
        returns1 = self.asset_returns[asset1]
        returns2 = self.asset_returns[asset2]
        
        # Calculate beta (hedge ratio)
        covariance = np.cov(returns1, returns2)[0, 1]
        variance = np.var(returns2)
        
        if variance > 0:
            hedge_ratio = covariance / variance
        else:
            hedge_ratio = 1.0
        
        return abs(hedge_ratio)
    
    def _estimate_p_value(self, correlation: float, sample_size: int) -> float:
        """
        Estimate p-value for correlation coefficient
        
        Args:
            correlation: Correlation coefficient
            sample_size: Number of samples
        
        Returns:
            Estimated p-value
        """
        if sample_size < 3:
            return 1.0
        
        # Fisher transformation
        import math
        from scipy import stats
        
        try:
            z = 0.5 * math.log((1 + correlation) / (1 - correlation))
            z_score = z * math.sqrt(sample_size - 3)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            return min(1.0, p_value)
        except:
            return 1.0
    
    def get_correlation_report(self) -> str:
        """Generate correlation analysis report"""
        if self.correlation_matrix is None:
            return "Correlation matrix not available"
        
        report = []
        report.append("=" * 60)
        report.append("CORRELATION ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Get upper triangle of correlation matrix
        upper_tri = self.correlation_matrix.where(
            np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find highest correlations
        high_corrs = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                val = upper_tri.loc[idx, col]
                if not np.isnan(val) and abs(val) > self.max_correlation:
                    high_corrs.append((idx, col, val))
        
        high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        report.append(f"\nHigh Correlations (> {self.max_correlation}):")
        for asset1, asset2, corr in high_corrs[:10]:
            report.append(f"  {asset1} <-> {asset2}: {corr:.3f}")
        
        # Portfolio risk metrics
        if len(self.asset_returns) > 0:
            avg_corr = self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)].mean()
            report.append(f"\nAverage Correlation: {avg_corr:.3f}")
            report.append(f"Total Assets: {len(self.correlation_matrix)}")
        
        report.append("=" * 60)
        
        return "\n".join(report)