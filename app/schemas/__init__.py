"""
__init__.py
Part of the app/schemas module.
Exports Pydantic schemas for request/response validation.
"""

from .trade import (
    TradeRequest, TradeResponse, OrderRequest, OrderResponse,
    Position, PositionResponse, TradeSignal
)
from .backtest import (
    BacktestRequest, BacktestResponse, BacktestResult,
    WalkForwardRequest, MonteCarloRequest, BacktestMetrics
)
from .analysis import (
    TechnicalAnalysisRequest, TechnicalAnalysisResponse,
    SentimentAnalysisRequest, SentimentAnalysisResponse,
    CorrelationRequest, CorrelationResponse, RegimeDetectionResponse
)
from .response import (
    APIResponse, PaginatedResponse, ErrorResponse,
    HealthResponse, MetricsResponse, StatusResponse
)

__all__ = [
    # Trade schemas
    'TradeRequest',
    'TradeResponse',
    'OrderRequest',
    'OrderResponse',
    'Position',
    'PositionResponse',
    'TradeSignal',
    
    # Backtest schemas
    'BacktestRequest',
    'BacktestResponse',
    'BacktestResult',
    'WalkForwardRequest',
    'MonteCarloRequest',
    'BacktestMetrics',
    
    # Analysis schemas
    'TechnicalAnalysisRequest',
    'TechnicalAnalysisResponse',
    'SentimentAnalysisRequest',
    'SentimentAnalysisResponse',
    'CorrelationRequest',
    'CorrelationResponse',
    'RegimeDetectionResponse',
    
    # Response schemas
    'APIResponse',
    'PaginatedResponse',
    'ErrorResponse',
    'HealthResponse',
    'MetricsResponse',
    'StatusResponse'
]