"""
__init__.py
Backtesting module — exports all public components.
"""

from .engine import BacktestEngine, Order, Trade, OrderType, OrderSide
from .metrics import (
    calculate_metrics,
    calculate_bootstrap_metrics,
    PerformanceMetrics,
    RiskMetrics,
)
from .monte_carlo import MonteCarloSimulator, SimulationResult
from .stress_tests import StressTester, StressScenario, MarketCrashScenario
from .walk_forward import WalkForwardValidator
from .visualizer import BacktestVisualizer, ChartGenerator

__all__ = [
    "BacktestEngine", "Order", "Trade", "OrderType", "OrderSide",
    "calculate_metrics", "calculate_bootstrap_metrics",
    "PerformanceMetrics", "RiskMetrics",
    "MonteCarloSimulator", "SimulationResult",
    "StressTester", "StressScenario", "MarketCrashScenario",
    "WalkForwardValidator",
    "BacktestVisualizer", "ChartGenerator",
]
