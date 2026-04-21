"""
monte_carlo.py
Part of the app/backtesting module.
Monte Carlo simulation for robust performance analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results"""
    mean_final_value: float
    median_final_value: float
    std_final_value: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    success_rate: float
    percentiles: Dict[int, float]
    all_paths: np.ndarray
    summary_stats: Dict[str, float]


class MonteCarloSimulator:
    """
    Monte Carlo simulator for trading strategies
    Simulates thousands of possible future scenarios
    """
    
    def __init__(self, n_simulations: int = 10000, n_days: int = 252):
        """
        Initialize Monte Carlo simulator
        
        Args:
            n_simulations: Number of simulations to run
            n_days: Number of days to simulate
        """
        self.n_simulations = n_simulations
        self.n_days = n_days
        self.results = None
        
    def simulate_normal(self, returns: pd.Series, initial_capital: float = 100000) -> SimulationResult:
        """
        Simulate using normal distribution (geometric Brownian motion)
        
        Args:
            returns: Historical returns series
            initial_capital: Starting capital
        
        Returns:
            SimulationResult object
        """
        # Calculate parameters
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate random returns
        dt = 1 / 252
        random_shocks = np.random.normal(mu * dt, sigma * np.sqrt(dt), 
                                         size=(self.n_simulations, self.n_days))
        
        # Simulate price paths
        paths = np.zeros((self.n_simulations, self.n_days + 1))
        paths[:, 0] = initial_capital
        
        for t in range(1, self.n_days + 1):
            paths[:, t] = paths[:, t-1] * (1 + random_shocks[:, t-1])
        
        return self._calculate_results(paths, initial_capital)
    
    def simulate_bootstrap(self, returns: pd.Series, initial_capital: float = 100000) -> SimulationResult:
        """
        Simulate using bootstrap resampling of historical returns
        
        Args:
            returns: Historical returns series
            initial_capital: Starting capital
        
        Returns:
            SimulationResult object
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
    
    def simulate_garch(self, returns: pd.Series, initial_capital: float = 100000) -> SimulationResult:
        """
        Simulate using GARCH(1,1) model for volatility clustering
        
        Args:
            returns: Historical returns series
            initial_capital: Starting capital
        
        Returns:
            SimulationResult object
        """
        try:
            from arch import arch_model
        except ImportError as exc:
            raise ImportError(
                "simulate_garch() requires the 'arch' package. "
                "Install it with: pip install arch"
            ) from exc
        
        # Fit GARCH model
        model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
        fitted = model.fit(disp='off')
        
        # Forecast volatility
        forecasts = fitted.forecast(horizon=self.n_days)
        simulated_vol = np.sqrt(forecasts.variance.values[-1]) / 100
        
        # Simulate returns with GARCH volatility
        mu = returns.mean()
        simulated_returns = np.random.normal(mu, simulated_vol, 
                                            size=(self.n_simulations, self.n_days))
        
        # Simulate paths
        paths = np.zeros((self.n_simulations, self.n_days + 1))
        paths[:, 0] = initial_capital
        
        for t in range(1, self.n_days + 1):
            paths[:, t] = paths[:, t-1] * (1 + simulated_returns[:, t-1])
        
        return self._calculate_results(paths, initial_capital)
    
    def simulate_with_regime_switching(self, returns: pd.Series, 
                                        initial_capital: float = 100000) -> SimulationResult:
        """
        Simulate with regime switching (bull/bear markets)
        
        Args:
            returns: Historical returns series
            initial_capital: Starting capital
        
        Returns:
            SimulationResult object
        """
        # Simple regime detection
        rolling_mean = returns.rolling(20).mean()
        bull_regime = rolling_mean > 0
        
        # Parameters for each regime
        bull_returns = returns[bull_regime].dropna()
        bear_returns = returns[~bull_regime].dropna()
        
        if len(bull_returns) == 0 or len(bear_returns) == 0:
            # Fallback to bootstrap
            return self.simulate_bootstrap(returns, initial_capital)
        
        # Regime switching probabilities
        bull_to_bull = 0.95
        bull_to_bear = 0.05
        bear_to_bear = 0.90
        bear_to_bull = 0.10
        
        # Simulate regime path
        regimes = np.zeros(self.n_days)
        current_regime = 0  # 0 = bull, 1 = bear
        
        for i in range(self.n_days):
            if current_regime == 0:  # Bull
                if np.random.random() < bull_to_bear:
                    current_regime = 1
            else:  # Bear
                if np.random.random() < bear_to_bull:
                    current_regime = 0
            regimes[i] = current_regime
        
        # Generate returns based on regime
        simulated_returns = np.zeros((self.n_simulations, self.n_days))
        
        for sim in range(self.n_simulations):
            for day in range(self.n_days):
                if regimes[day] == 0:  # Bull
                    ret = np.random.choice(bull_returns)
                else:  # Bear
                    ret = np.random.choice(bear_returns)
                simulated_returns[sim, day] = ret
        
        # Simulate paths
        paths = np.zeros((self.n_simulations, self.n_days + 1))
        paths[:, 0] = initial_capital
        
        for t in range(1, self.n_days + 1):
            paths[:, t] = paths[:, t-1] * (1 + simulated_returns[:, t-1])
        
        return self._calculate_results(paths, initial_capital)
    
    def _calculate_results(self, paths: np.ndarray, initial_capital: float) -> SimulationResult:
        """Calculate statistics from simulated paths"""
        final_values = paths[:, -1]
        
        # Sort final values
        sorted_final = np.sort(final_values)
        
        # Calculate percentiles
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
        
        # Calculate VaR and CVaR
        var_95 = initial_capital - np.percentile(final_values, 5)
        var_99 = initial_capital - np.percentile(final_values, 1)
        
        cvar_95 = initial_capital - final_values[final_values <= np.percentile(final_values, 5)].mean()
        cvar_99 = initial_capital - final_values[final_values <= np.percentile(final_values, 1)].mean()
        
        # Success rate (ending with more than initial capital)
        success_rate = np.mean(final_values > initial_capital)
        
        # Summary statistics
        summary_stats = {
            'mean_return': (final_values.mean() / initial_capital - 1) * 100,
            'median_return': (np.median(final_values) / initial_capital - 1) * 100,
            'std_return': (final_values.std() / initial_capital) * 100,
            'best_case': (final_values.max() / initial_capital - 1) * 100,
            'worst_case': (final_values.min() / initial_capital - 1) * 100,
            'downside_risk': np.mean(final_values < initial_capital) * 100
        }
        
        self.results = SimulationResult(
            mean_final_value=final_values.mean(),
            median_final_value=np.median(final_values),
            std_final_value=final_values.std(),
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            success_rate=success_rate * 100,
            percentiles=percentiles,
            all_paths=paths,
            summary_stats=summary_stats
        )
        
        return self.results
    
    def calculate_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for final value"""
        if self.results is None:
            raise ValueError("Run simulation first")
        
        lower = (1 - confidence) / 2
        upper = 1 - lower
        
        return (np.percentile(self.results.all_paths[:, -1], lower * 100),
                np.percentile(self.results.all_paths[:, -1], upper * 100))
    
    def probability_of_loss(self, loss_threshold: float = 0) -> float:
        """Calculate probability of loss exceeding threshold"""
        if self.results is None:
            raise ValueError("Run simulation first")
        
        final_values = self.results.all_paths[:, -1]
        return np.mean(final_values < loss_threshold) * 100
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get comprehensive risk metrics from simulation"""
        if self.results is None:
            raise ValueError("Run simulation first")
        
        final_values = self.results.all_paths[:, -1]
        
        return {
            'probability_of_ruin': self.probability_of_loss(0),
            'expected_shortfall': self.results.cvar_95,
            'value_at_risk': self.results.var_95,
            'worst_case_10pct': self.results.percentiles[10],
            'best_case_90pct': self.results.percentiles[90],
            'upside_potential': (self.results.percentiles[90] - self.results.median_final_value) / self.results.median_final_value * 100,
            'downside_risk': (self.results.median_final_value - self.results.percentiles[10]) / self.results.median_final_value * 100
        }