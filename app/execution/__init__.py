"""
__init__.py
Part of the app/execution module.
Exports order execution components for smart routing and algorithmic execution.
"""

from .order_executor import OrderExecutor, Order, OrderStatus, OrderType, OrderSide
from .twap_executor import TWAPExecutor, TWAPOrder
from .smart_routing import SmartRouter, RoutingDecision, Exchange
from .feedback_loop import ExecutionFeedbackLoop, ExecutionMetrics

__all__ = [
    'OrderExecutor',
    'Order',
    'OrderStatus',
    'OrderType',
    'OrderSide',
    'TWAPExecutor',
    'TWAPOrder',
    'SmartRouter',
    'RoutingDecision',
    'Exchange',
    'ExecutionFeedbackLoop',
    'ExecutionMetrics'
]