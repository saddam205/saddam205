"""
correlation.py
Part of the app/analysis module.
Analyzes correlations between multiple assets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class CorrelationResult:
    """Container for correlation analysis results"""
    correlation_matrix: pd.DataFrame
    p_values: pd.DataFrame
    top_correlations: List[Tuple[str, str, float]]
    hedge_opportunities: List[Dict]


class CorrelationAnalyzer:
    """Analyzes correlations between multiple assets"""
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame], lookback: int = 100):
        """
        Initialize correlation analyzer
        
        Args:
            data_dict: Dictionary mapping asset symbols to their DataFrames
            lookback: Lookback period for correlation calculation
        """
        self.data_dict = data_dict
        self.lookback = lookback
        self.returns = {}
        
        self._calculate_returns()
        
    def _calculate_returns(self):
        """Calculate returns for all assets"""
        for symbol, data in self.data_dict.items():
            if 'close' in data.columns:
                self.returns[symbol] = data['close'].pct_change().dropna()
    
    def compute_correlations(self, method: str = 'pearson') -> CorrelationResult:
        """
        Compute correlation matrix between all assets
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
        
        Returns:
            CorrelationResult object
        """
        # Create returns DataFrame
        returns_df = pd.DataFrame(self.returns)
        
        # Get latest lookback period
        if len(returns_df) > self.lookback:
            returns_df = returns_df.tail(self.lookback)
        
        # Compute correlations
        corr_matrix = returns_df.corr(method=method)
        
        # Compute p-values
        p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
        
        # Select the correct statistical test to match the correlation method
        _pval_func = {
            'pearson':  lambda a, b: stats.pearsonr(a, b)[1],
            'spearman': lambda a, b: stats.spearmanr(a, b)[1],
            'kendall':  lambda a, b: stats.kendalltau(a, b)[1],
        }.get(method, lambda a, b: stats.pearsonr(a, b)[1])

        for i in corr_matrix.index:
            for j in corr_matrix.columns:
                if i != j:
                    p_value = _pval_func(
                        returns_df[i].dropna(),
                        returns_df[j].dropna()
                    )
                    p_values.loc[i, j] = p_value
                else:
                    p_values.loc[i, j] = 0
        
        # Find top correlations
        top_correlations = self._get_top_correlations(corr_matrix)
        
        # Identify hedge opportunities
        hedge_opps = self._find_hedge_opportunities(corr_matrix)
        
        return CorrelationResult(
            correlation_matrix=corr_matrix,
            p_values=p_values,
            top_correlations=top_correlations,
            hedge_opportunities=hedge_opps
        )
    
    def _get_top_correlations(self, corr_matrix: pd.DataFrame, n: int = 10) -> List[Tuple[str, str, float]]:
        """Get top N correlations"""
        correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if not np.isnan(corr_value):
                    correlations.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return correlations[:n]
    
    def _find_hedge_opportunities(self, corr_matrix: pd.DataFrame) -> List[Dict]:
        """Find potential hedging opportunities"""
        opportunities = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                
                # Strong negative correlation indicates good hedge
                if corr < -0.7:
                    opportunities.append({
                        'asset1': corr_matrix.columns[i],
                        'asset2': corr_matrix.columns[j],
                        'correlation': corr,
                        'hedge_ratio': 1 / abs(corr),
                        'type': 'negative_correlation'
                    })
                # Low correlation also provides diversification
                elif abs(corr) < 0.2:
                    opportunities.append({
                        'asset1': corr_matrix.columns[i],
                        'asset2': corr_matrix.columns[j],
                        'correlation': corr,
                        'hedge_ratio': 1,
                        'type': 'diversification'
                    })
        
        return opportunities
    
    def compute_rolling_correlations(self, window: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Compute rolling correlations
        
        Args:
            window: Rolling window size
        
        Returns:
            Dictionary of rolling correlation DataFrames
        """
        rolling_corrs = {}
        returns_df = pd.DataFrame(self.returns)
        
        for i in range(len(returns_df.columns)):
            for j in range(i + 1, len(returns_df.columns)):
                asset1 = returns_df.columns[i]
                asset2 = returns_df.columns[j]
                
                rolling_corr = returns_df[asset1].rolling(window).corr(returns_df[asset2])
                rolling_corrs[f"{asset1}_{asset2}"] = rolling_corr
        
        return rolling_corrs
    
    def compute_beta(self, market_index: str) -> Dict[str, float]:
        """
        Compute beta relative to market index
        
        Args:
            market_index: Symbol of market index asset
        
        Returns:
            Dictionary mapping assets to beta values
        """
        if market_index not in self.returns:
            raise ValueError(f"Market index {market_index} not found")
        
        market_returns = self.returns[market_index]
        betas = {}
        
        for symbol, returns in self.returns.items():
            if symbol != market_index:
                # Calculate covariance and variance
                covariance = np.cov(returns, market_returns, ddof=0)[0, 1]
                variance = np.var(market_returns, ddof=0)
                beta = covariance / variance if variance > 0 else 0
                betas[symbol] = beta
        
        return betas
    
    def compute_correlation_network(self, threshold: float = 0.5) -> Dict:
        """
        Build correlation network graph
        
        Args:
            threshold: Minimum correlation to include in network
        
        Returns:
            Dictionary with nodes and edges
        """
        corr_matrix = self.compute_correlations().correlation_matrix
        
        nodes = [{'id': symbol, 'label': symbol} for symbol in corr_matrix.index]
        edges = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    edges.append({
                        'from': corr_matrix.columns[i],
                        'to': corr_matrix.columns[j],
                        'weight': abs(corr),
                        'sign': 'positive' if corr > 0 else 'negative'
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'density': len(edges) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0
        }
    
    def detect_correlation_breakdown(self, recent_window: int = 20, 
                                      historical_window: int = 100) -> Dict[str, float]:
        """
        Detect breakdown in historical correlations
        
        Args:
            recent_window: Recent period window
            historical_window: Historical period window
        
        Returns:
            Dictionary with correlation change metrics
        """
        returns_df = pd.DataFrame(self.returns)
        
        breakdowns = {}
        
        for i in range(len(returns_df.columns)):
            for j in range(i + 1, len(returns_df.columns)):
                asset1 = returns_df.columns[i]
                asset2 = returns_df.columns[j]
                
                # Historical correlation
                hist_corr = returns_df[asset1].tail(historical_window).corr(returns_df[asset2])
                
                # Recent correlation
                recent_corr = returns_df[asset1].tail(recent_window).corr(returns_df[asset2])
                
                # Change in correlation
                corr_change = abs(recent_corr - hist_corr)
                
                if corr_change > 0.5:
                    breakdowns[f"{asset1}_{asset2}"] = {
                        'historical_correlation': hist_corr,
                        'recent_correlation': recent_corr,
                        'change': corr_change,
                        'breakdown_severity': 'high' if corr_change > 0.7 else 'medium'
                    }
        
        return breakdowns