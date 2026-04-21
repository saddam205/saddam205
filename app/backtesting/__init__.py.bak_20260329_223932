"""
metrics.py
Part of the app/backtesting module.
Comprehensive performance metrics for backtesting analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Container for comprehensive performance metrics"""
    # Returns
    total_return: float
    annualized_return: float
    monthly_return: float
    weekly_return: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    value_at_risk: float
    conditional_var: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    best_trade: float
    worst_trade: float
    
    # Drawdown analysis
    drawdowns: List[Dict]
    recovery_factor: float
    
    # Rolling metrics
    rolling_sharpe: pd.Series
    rolling_volatility: pd.Series


def calculate_metrics(returns: pd.Series, trades: List = None, 
                      benchmark_returns: pd.Series = None) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics
    
    Args:
        returns: Series of returns (percentage or decimal)
        trades: List of trade objects with PnL
        benchmark_returns: Benchmark returns for comparison
    
    Returns:
        PerformanceMetrics object with all calculations
    """
    # Clean returns
    returns = returns.dropna()
    
    if len(returns) < 2:
        raise ValueError("Insufficient return data for metric calculation")
    
    # Basic statistics
    total_return = (1 + returns).prod() - 1
    
    # Annualized metrics
    years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annualized_volatility = returns.std() * np.sqrt(252)
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()
    max_drawdown_duration = _calculate_drawdown_duration(drawdowns)
    
    # Risk metrics
    value_at_risk = returns.quantile(0.05)
    conditional_var = returns[returns <= value_at_risk].mean()
    
    # Risk-adjusted ratios
    sharpe_ratio = (annualized_return - 0.02) / annualized_volatility if annualized_volatility > 0 else 0
    
    # Sortino Ratio (uses downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annualized_volatility
    sortino_ratio = (annualized_return - 0.02) / downside_deviation if downside_deviation > 0 else 0
    
    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # Omega Ratio
    omega_ratio = _calculate_omega_ratio(returns, threshold=0)
    
    # Trade statistics
    if trades:
        trade_pnls = [t.pnl for t in trades] if hasattr(trades[0], 'pnl') else trades
        total_trades = len(trade_pnls)
        winning_trades = [p for p in trade_pnls if p > 0]
        losing_trades = [p for p in trade_pnls if p <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        gross_profit = sum(winning_trades)
        gross_loss = abs(sum(losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        best_trade = max(trade_pnls)
        worst_trade = min(trade_pnls)
    else:
        total_trades = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        expectancy = 0
        best_trade = 0
        worst_trade = 0
    
    # Distribution metrics
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Recovery factor
    recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Rolling metrics
    rolling_sharpe = returns.rolling(252).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
    )
    rolling_volatility = returns.rolling(20).std() * np.sqrt(252)
    
    # Monthly and weekly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    weekly_returns = returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
    
    # Drawdown analysis
    drawdown_list = _analyze_drawdowns(drawdowns)
    
    return PerformanceMetrics(
        total_return=total_return * 100,
        annualized_return=annualized_return * 100,
        monthly_return=monthly_returns.mean() * 100,
        weekly_return=weekly_returns.mean() * 100,
        volatility=annualized_volatility * 100,
        max_drawdown=max_drawdown * 100,
        max_drawdown_duration=max_drawdown_duration,
        value_at_risk=value_at_risk * 100,
        conditional_var=conditional_var * 100,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        omega_ratio=omega_ratio,
        total_trades=total_trades,
        winning_trades=len(winning_trades) if trades else 0,
        losing_trades=len(losing_trades) if trades else 0,
        win_rate=win_rate * 100,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        skewness=skewness,
        kurtosis=kurtosis,
        best_trade=best_trade,
        worst_trade=worst_trade,
        drawdowns=drawdown_list,
        recovery_factor=recovery_factor,
        rolling_sharpe=rolling_sharpe,
        rolling_volatility=rolling_volatility
    )


def _calculate_drawdown_duration(drawdowns: pd.Series) -> int:
    """Calculate maximum drawdown duration in days"""
    is_drawdown = drawdowns < 0
    max_duration = 0
    current_duration = 0
    
    for is_dd in is_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return max_duration


def _analyze_drawdowns(drawdowns: pd.Series) -> List[Dict]:
    """Analyze individual drawdown periods"""
    drawdown_periods = []
    in_drawdown = False
    start_idx = None
    max_dd = 0
    
    for i, dd in enumerate(drawdowns):
        if dd < 0 and not in_drawdown:
            in_drawdown = True
            start_idx = i
            max_dd = dd
        elif dd < 0 and in_drawdown:
            max_dd = min(max_dd, dd)
        elif dd == 0 and in_drawdown:
            drawdown_periods.append({
                'start': start_idx,
                'end': i,
                'depth': max_dd * 100,
                'duration': i - start_idx
            })
            in_drawdown = False
    
    return drawdown_periods


def _calculate_omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
    """Calculate Omega ratio (probability-weighted ratio of gains to losses)"""
    returns_above = returns[returns > threshold] - threshold
    returns_below = threshold - returns[returns < threshold]
    
    if len(returns_below) == 0 or returns_below.sum() == 0:
        return float('inf')
    
    return returns_above.sum() / returns_below.sum()


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_shortfall: float
    maximum_loss: float
    average_loss: float
    worst_case_scenario: float
    
    @classmethod
    def calculate(cls, returns: pd.Series) -> 'RiskMetrics':
        """Calculate risk metrics from returns"""
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        negative_returns = returns[returns < 0]
        expected_shortfall = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        return cls(
            var_95=var_95 * 100,
            var_99=var_99 * 100,
            cvar_95=cvar_95 * 100,
            cvar_99=cvar_99 * 100,
            expected_shortfall=expected_shortfall * 100,
            maximum_loss=negative_returns.min() * 100,
            average_loss=negative_returns.mean() * 100,
            worst_case_scenario=negative_returns.quantile(0.01) * 100
        )


def calculate_bootstrap_metrics(returns: pd.Series, n_iterations: int = 1000) -> Dict:
    """
    Calculate metrics with bootstrap confidence intervals
    
    Args:
        returns: Return series
        n_iterations: Number of bootstrap iterations
    
    Returns:
        Dictionary with metrics and confidence intervals
    """
    n_samples = len(returns)
    bootstrap_metrics = []
    
    for _ in range(n_iterations):
        # Sample with replacement
        sample = np.random.choice(returns, size=n_samples, replace=True)
        metrics = calculate_metrics(pd.Series(sample))
        bootstrap_metrics.append({
            'sharpe': metrics.sharpe_ratio,
            'sortino': metrics.sortino_ratio,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate
        })
    
    # Calculate confidence intervals
    results = {}
    for metric in ['sharpe', 'sortino', 'max_drawdown', 'win_rate']:
        values = [m[metric] for m in bootstrap_metrics]
        results[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_95_lower': np.percentile(values, 2.5),
            'ci_95_upper': np.percentile(values, 97.5),
            'ci_99_lower': np.percentile(values, 0.5),
            'ci_99_upper': np.percentile(values, 99.5)
        }
    
    return results