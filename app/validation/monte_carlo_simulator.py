"""
monte_carlo_simulator.py
Part of the app/validation module.
Monte Carlo simulation for robust performance analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results"""
    mean_final_value: float
    median_final_value: float
    std_final_value: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    success_rate: float
    percentiles: Dict[int, float]
    worst_case: float
    best_case: float


class MonteCarloSimulator:
    """
    Monte Carlo simulator for trading strategies
    """
    
    def __init__(self, n_simulations: int = 10000, n_days: int = 252):
        """
        Initialize Monte Carlo simulator
        
        Args:
            n_simulations: Number of simulations
            n_days: Number of days to simulate
        """
        self.n_simulations = n_simulations
        self.n_days = n_days
        self.results: Optional[MonteCarloResult] = None
        
    def simulate_from_returns(self, returns: np.ndarray, 
                             initial_capital: float = 100000) -> MonteCarloResult:
        """
        Simulate from historical returns
        
        Args:
            returns: Historical return series
            initial_capital: Starting capital
        
        Returns:
            Monte Carlo results
        """
        # Bootstrap returns
        n_historical = len(returns)
        bootstrap_returns = np.random.choice(returns, 
                                            size=(self.n_simulations, self.n_days),
                                            replace=True)
        
        # Simulate paths
        paths = np.zeros((self.n_simulations, self.n_days + 1))
        paths[:, 0] = initial_capital
        
        for t in range(1, self.n_days + 1):
            paths[:, t] = paths[:, t-1] * (1 + bootstrap_returns[:, t-1])
        
        return self._calculate_results(paths, initial_capital)
    
    def simulate_normal(self, mu: float, sigma: float,
                       initial_capital: float = 100000) -> MonteCarloResult:
        """
        Simulate using normal distribution
        
        Args:
            mu: Mean return
            sigma: Standard deviation
            initial_capital: Starting capital
        
        Returns:
            Monte Carlo results
        """
        dt = 1 / 252
        random_shocks = np.random.normal(mu * dt, sigma * np.sqrt(dt), 
                                         size=(self.n_simulations, self.n_days))
        
        paths = np.zeros((self.n_simulations, self.n_days + 1))
        paths[:, 0] = initial_capital
        
        for t in range(1, self.n_days + 1):
            paths[:, t] = paths[:, t-1] * (1 + random_shocks[:, t-1])
        
        return self._calculate_results(paths, initial_capital)
    
    def simulate_geometric_brownian(self, mu: float, sigma: float,
                                    initial_capital: float = 100000) -> MonteCarloResult:
        """
        Simulate using Geometric Brownian Motion
        
        Args:
            mu: Drift
            sigma: Volatility
            initial_capital: Starting capital
        
        Returns:
            Monte Carlo results
        """
        dt = 1 / 252
        paths = np.zeros((self.n_simulations, self.n_days + 1))
        paths[:, 0] = initial_capital
        
        for t in range(1, self.n_days + 1):
            random_shocks = np.random.normal(0, 1, self.n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks
            )
        
        return self._calculate_results(paths, initial_capital)
    
    def _calculate_results(self, paths: np.ndarray, 
                          initial_capital: float) -> MonteCarloResult:
        """Calculate statistics from simulated paths"""
        final_values = paths[:, -1]
        sorted_final = np.sort(final_values)
        
        percentiles = {
            1: np.percentile(final_values, 1),
            5: np.percentile(final_values, 5),
            10: np.percentile(final_values, 10),
            25: np.percentile(final_values, 25),
            50: np.percentile(final_values, 50),
            75: np.percentile(final_values, 75),
            90: np.percentile(final_values, 90),
            95: np.percentile(final_values, 95),
            99: np.percentile(final_values, 99)
        }
        
        var_95 = initial_capital - np.percentile(final_values, 5)
        var_99 = initial_capital - np.percentile(final_values, 1)
        
        cvar_95 = initial_capital - final_values[final_values <= np.percentile(final_values, 5)].mean()
        cvar_99 = initial_capital - final_values[final_values <= np.percentile(final_values, 1)].mean()
        
        success_rate = np.mean(final_values > initial_capital) * 100
        
        self.results = MonteCarloResult(
            mean_final_value=final_values.mean(),
            median_final_value=np.median(final_values),
            std_final_value=final_values.std(),
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            success_rate=success_rate,
            percentiles=percentiles,
            worst_case=final_values.min(),
            best_case=final_values.max()
        )
        
        return self.results
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for final value"""
        if self.results is None:
            raise ValueError("Run simulation first")
        
        lower = (1 - confidence) / 2
        upper = 1 - lower
        
        return (self.results.percentiles[int(lower * 100)],
                self.results.percentiles[int(upper * 100)])
    
    def probability_of_ruin(self, ruin_threshold: float = 0) -> float:
        """Calculate probability of ruin"""
        if self.results is None:
            raise ValueError("Run simulation first")
        
        # Need access to paths
        return 0  # Placeholder
    
    def get_summary(self) -> Dict:
        """Get simulation summary"""
        if self.results is None:
            return {'message': 'No simulation results'}
        
        return {
            'mean_final_value': self.results.mean_final_value,
            'median_final_value': self.results.median_final_value,
            'std_final_value': self.results.std_final_value,
            'success_rate': self.results.success_rate,
            'var_95': self.results.var_95,
            'var_99': self.results.var_99,
            'cvar_95': self.results.cvar_95,
            'cvar_99': self.results.cvar_99,
            'best_case': self.results.best_case,
            'worst_case': self.results.worst_case
        }