"""
backtest.py
Part of the app/schemas module.
Pydantic schemas for backtesting operations.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class StrategyType(str, Enum):
    """Strategy types"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    CUSTOM = "custom"


class BacktestInterval(str, Enum):
    """Backtest time intervals"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class BacktestRequest(BaseModel):
    """Backtest execution request"""
    symbol: str = Field(..., description="Trading symbol")
    strategy: StrategyType = Field(..., description="Strategy to test")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    interval: BacktestInterval = Field(BacktestInterval.HOUR_1, description="Time interval")
    initial_capital: float = Field(100000, gt=0, description="Initial capital")
    commission: float = Field(0.001, ge=0, le=0.01, description="Commission rate")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Strategy parameters")
    
    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('End date must be after start date')
        return v


class BacktestMetrics(BaseModel):
    """Backtest performance metrics"""
    total_return: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return")
    volatility: float = Field(..., description="Annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Win rate percentage")
    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    profit_factor: float = Field(..., description="Profit factor (gross profit / gross loss)")
    expectancy: float = Field(..., description="Average profit per trade")
    avg_win: float = Field(..., description="Average winning trade")
    avg_loss: float = Field(..., description="Average losing trade")
    best_trade: float = Field(..., description="Best trade P&L")
    worst_trade: float = Field(..., description="Worst trade P&L")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    recovery_factor: float = Field(..., description="Recovery factor")


class BacktestTrade(BaseModel):
    """Individual backtest trade"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    pnl: float
    pnl_percent: float
    holding_hours: float


class BacktestResult(BaseModel):
    """Complete backtest result"""
    request: BacktestRequest
    metrics: BacktestMetrics
    trades: List[BacktestTrade] = Field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = Field(default_factory=list)
    monthly_returns: Dict[str, float] = Field(default_factory=dict)
    start_date: datetime
    end_date: datetime
    execution_time_seconds: float
    final_capital: float
    
    @property
    def total_trades_count(self) -> int:
        return len(self.trades)


class BacktestResponse(BaseModel):
    """Backtest API response"""
    success: bool
    message: str
    result: Optional[BacktestResult] = None
    error: Optional[str] = None


class WalkForwardRequest(BaseModel):
    """Walk-forward validation request"""
    symbol: str = Field(..., description="Trading symbol")
    strategy: StrategyType = Field(..., description="Strategy to test")
    start_date: datetime = Field(..., description="Start date")
    end_date: datetime = Field(..., description="End date")
    n_splits: int = Field(5, ge=2, le=20, description="Number of forward windows")
    train_ratio: float = Field(0.6, ge=0.5, le=0.8, description="Training ratio")
    val_ratio: float = Field(0.2, ge=0.1, le=0.3, description="Validation ratio")
    interval: BacktestInterval = Field(BacktestInterval.HOUR_1)
    initial_capital: float = Field(100000, gt=0)
    
    @validator('train_ratio', 'val_ratio')
    def validate_ratios(cls, v, values):
        if 'train_ratio' in values and values.get('train_ratio', 0) + values.get('val_ratio', 0) > 0.95:
            raise ValueError('Train + val ratio must be <= 0.95')
        return v


class WalkForwardWindow(BaseModel):
    """Single walk-forward window result"""
    window_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    metrics: BacktestMetrics
    trades_count: int


class WalkForwardResult(BaseModel):
    """Walk-forward validation result"""
    windows: List[WalkForwardWindow]
    aggregate_metrics: BacktestMetrics
    robustness_score: float = Field(..., ge=0, le=1)
    stability_score: float = Field(..., ge=0, le=1)
    is_robust: bool


class MonteCarloRequest(BaseModel):
    """Monte Carlo simulation request"""
    symbol: str = Field(..., description="Trading symbol")
    n_simulations: int = Field(1000, ge=100, le=100000, description="Number of simulations")
    n_days: int = Field(252, ge=30, le=1000, description="Simulation days")
    initial_capital: float = Field(100000, gt=0)
    simulation_type: str = Field("bootstrap", description="normal, bootstrap, garch")
    confidence_level: float = Field(0.95, ge=0.9, le=0.99)


class MonteCarloResult(BaseModel):
    """Monte Carlo simulation result"""
    mean_final_value: float
    median_final_value: float
    std_final_value: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    success_rate: float = Field(..., description="Probability of profit")
    probability_of_loss: float = Field(..., description="Probability of loss")
    best_case: float
    worst_case: float
    confidence_interval_lower: float
    confidence_interval_upper: float


class StressScenario(BaseModel):
    """Stress test scenario"""
    name: str = Field(..., description="Scenario name")
    market_shock: float = Field(..., description="Market shock percentage")
    volatility_multiplier: float = Field(1.0, description="Volatility multiplier")
    correlation_change: float = Field(0, description="Correlation change")
    liquidity_multiplier: float = Field(1.0, description="Liquidity multiplier")


class StressTestRequest(BaseModel):
    """Stress test request"""
    symbol: str = Field(..., description="Trading symbol")
    strategy: StrategyType = Field(..., description="Strategy to test")
    scenarios: List[StressScenario] = Field(..., description="Scenarios to test")
    initial_capital: float = Field(100000, gt=0)


class StressTestResult(BaseModel):
    """Stress test result"""
    scenario_name: str
    survived: bool
    final_capital: float
    max_drawdown: float
    loss_percent: float
    recovery_time_days: Optional[float] = None
    resilience_score: float = Field(..., ge=0, le=100)