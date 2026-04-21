"""
__init__.py
Part of the app/filters module.
Exports trade filtering components for signal validation and quality control.
"""

from .trade_filter import TradeFilter, FilterResult, FilterCondition
from .edge_detector import EdgeDetector, StatisticalEdge, EdgeType
from .correlation_filter import CorrelationFilter, AssetCorrelation

__all__ = [
    'TradeFilter',
    'FilterResult',
    'FilterCondition',
    'EdgeDetector',
    'StatisticalEdge',
    'EdgeType',
    'CorrelationFilter',
    'AssetCorrelation'
]