"""
order_executor.py
Part of the app/execution module.
Core order execution with support for multiple order types and exchanges.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(Enum):
    """Order sides"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order statuses"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """Order object"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_percent: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    client_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_completed(self) -> bool:
        """Check if order is completed"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    def fill(self, fill_price: float, fill_quantity: float) -> Dict:
        """Fill part of the order"""
        if fill_quantity > self.remaining_quantity:
            fill_quantity = self.remaining_quantity
        
        # Update filled quantity and average price
        old_filled = self.filled_quantity
        old_avg = self.avg_fill_price
        
        self.filled_quantity += fill_quantity
        self.avg_fill_price = ((old_avg * old_filled) + (fill_price * fill_quantity)) / self.filled_quantity
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        
        self.updated_at = datetime.now()
        
        return {
            'order_id': self.id,
            'fill_price': fill_price,
            'fill_quantity': fill_quantity,
            'remaining': self.remaining_quantity
        }
    
    def cancel(self):
        """Cancel order"""
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.now()
    
    def reject(self, reason: str):
        """Reject order"""
        self.status = OrderStatus.REJECTED
        self.metadata['reject_reason'] = reason
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'remaining_quantity': self.remaining_quantity
        }


class OrderExecutor:
    """
    Core order execution engine with support for multiple order types
    and exchange integration.
    """
    
    def __init__(self, mode: str = "VIRTUAL", slippage_model: str = "fixed"):
        """
        Initialize order executor
        
        Args:
            mode: Execution mode ('VIRTUAL' or 'REAL')
            slippage_model: Slippage model ('fixed', 'volume', 'volatility')
        """
        self.mode = mode
        self.slippage_model = slippage_model
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_volume': 0.0,
            'avg_slippage': 0.0
        }
        
        # Exchange connections (would be initialized with actual clients)
        self.exchanges = {}
        
    async def execute(self, symbol: str, side: str, quantity: float,
                     order_type: str = "MARKET", price: float = None,
                     stop_price: float = None, trail_percent: float = None,
                     timeout: int = 30, **kwargs) -> Dict:
        """
        Execute an order
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            order_type: Order type (MARKET, LIMIT, STOP_LOSS, etc.)
            price: Limit price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)
            trail_percent: Trail percentage (for TRAILING_STOP)
            timeout: Timeout in seconds
            **kwargs: Additional parameters
        
        Returns:
            Order execution result
        """
        # Create order object
        order = Order(
            id=f"{symbol}_{datetime.now().timestamp()}_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            order_type=OrderType(order_type.upper()),
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            trail_percent=trail_percent,
            client_order_id=kwargs.get('client_order_id')
        )
        
        self.orders[order.id] = order
        self.execution_stats['total_orders'] += 1
        
        logger.info(f"Created order: {order.side.value} {order.quantity} {order.symbol} ({order.order_type.value})")
        
        try:
            if self.mode == "VIRTUAL":
                result = await self._execute_virtual(order, **kwargs)
            else:
                result = await self._execute_real(order, **kwargs)
            
            # Update stats
            if result.get('status') == 'FILLED':
                self.execution_stats['filled_orders'] += 1
                self.execution_stats['total_volume'] += order.quantity * order.avg_fill_price
            elif result.get('status') == 'CANCELLED':
                self.execution_stats['cancelled_orders'] += 1
            elif result.get('status') == 'REJECTED':
                self.execution_stats['rejected_orders'] += 1
            
            # Add to history
            self.order_history.append(order)
            
            # Clean up completed orders
            if order.is_completed:
                del self.orders[order.id]
            
            return result
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            order.reject(str(e))
            self.order_history.append(order)
            del self.orders[order.id]
            
            return {
                'success': False,
                'error': str(e),
                'order_id': order.id,
                'status': 'REJECTED'
            }
    
    async def _execute_virtual(self, order: Order, **kwargs) -> Dict:
        """Execute order in virtual mode (paper trading)"""
        await asyncio.sleep(0.05)  # Simulate latency
        
        # Get current market price (would fetch from market data)
        current_price = kwargs.get('current_price', 100.0)
        
        # Calculate execution price based on order type
        if order.order_type == OrderType.MARKET:
            execution_price = self._calculate_slippage(current_price, order.quantity)
            fill_quantity = order.quantity
            
        elif order.order_type == OrderType.LIMIT:
            if order.price:
                execution_price = order.price
                # Check if limit order can be filled
                if (order.side == OrderSide.BUY and current_price <= order.price) or \
                   (order.side == OrderSide.SELL and current_price >= order.price):
                    fill_quantity = order.quantity
                else:
                    # Limit order not triggered
                    order.status = OrderStatus.PENDING
                    return {
                        'success': True,
                        'order_id': order.id,
                        'status': 'PENDING',
                        'message': 'Limit order pending'
                    }
            else:
                order.reject("Limit price required for LIMIT order")
                return {'success': False, 'error': 'Limit price required'}
            
        elif order.order_type == OrderType.STOP_LOSS:
            if order.stop_price:
                if (order.side == OrderSide.BUY and current_price >= order.stop_price) or \
                   (order.side == OrderSide.SELL and current_price <= order.stop_price):
                    execution_price = self._calculate_slippage(current_price, order.quantity)
                    fill_quantity = order.quantity
                else:
                    order.status = OrderStatus.PENDING
                    return {
                        'success': True,
                        'order_id': order.id,
                        'status': 'PENDING',
                        'message': 'Stop loss order pending'
                    }
            else:
                order.reject("Stop price required for STOP_LOSS order")
                return {'success': False, 'error': 'Stop price required'}
            
        else:
            order.reject(f"Unsupported order type: {order.order_type.value}")
            return {'success': False, 'error': 'Unsupported order type'}
        
        # Fill the order
        fill_result = order.fill(execution_price, fill_quantity)
        
        return {
            'success': True,
            'order_id': order.id,
            'status': order.status.value,
            'filled_quantity': order.filled_quantity,
            'avg_price': order.avg_fill_price,
            'slippage': execution_price - current_price if order.side == OrderSide.BUY else current_price - execution_price,
            'fill_details': fill_result
        }
    
    async def _execute_real(self, order: Order, **kwargs) -> Dict:
        """Execute order in real mode on exchange"""
        try:
            # Determine which exchange to use
            exchange = self._get_exchange(order.symbol)
            
            if not exchange:
                raise Exception(f"No exchange available for {order.symbol}")
            
            # Submit order to exchange
            exchange_order = await exchange.submit_order(
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price
            )
            
            # Update order with exchange data
            order.exchange_order_id = exchange_order.get('id')
            order.status = OrderStatus(exchange_order.get('status', 'SUBMITTED'))
            order.filled_quantity = exchange_order.get('filled_quantity', 0)
            order.avg_fill_price = exchange_order.get('avg_price', 0)
            
            return {
                'success': True,
                'order_id': order.id,
                'exchange_order_id': order.exchange_order_id,
                'status': order.status.value,
                'filled_quantity': order.filled_quantity,
                'avg_price': order.avg_fill_price
            }
            
        except Exception as e:
            logger.error(f"Real execution failed: {e}")
            order.reject(str(e))
            return {'success': False, 'error': str(e), 'order_id': order.id}
    
    def _calculate_slippage(self, price: float, quantity: float) -> float:
        """Calculate slippage based on order size"""
        if self.slippage_model == 'fixed':
            slippage_rate = 0.0005  # 0.05% fixed slippage
        elif self.slippage_model == 'volume':
            # Higher slippage for larger orders
            slippage_rate = min(0.01, quantity / 10000)  # Up to 1%
        else:
            slippage_rate = 0.001  # Default 0.1%
        
        return price * (1 + slippage_rate)
    
    def _get_exchange(self, symbol: str) -> Any:
        """Get appropriate exchange for symbol"""
        # This would return the correct exchange client
        return self.exchanges.get('default')
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an existing order"""
        if order_id not in self.orders:
            return {'success': False, 'error': 'Order not found'}
        
        order = self.orders[order_id]
        order.cancel()
        
        logger.info(f"Cancelled order: {order_id}")
        
        return {
            'success': True,
            'order_id': order_id,
            'status': 'CANCELLED'
        }
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order details"""
        if order_id in self.orders:
            return self.orders[order_id].to_dict()
        
        # Check history
        for order in self.order_history:
            if order.id == order_id:
                return order.to_dict()
        
        return None
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders"""
        open_orders = []
        for order in self.orders.values():
            if not order.is_completed:
                if symbol is None or order.symbol == symbol:
                    open_orders.append(order.to_dict())
        return open_orders
    
    def get_order_history(self, limit: int = 100, symbol: str = None) -> List[Dict]:
        """Get order history"""
        history = self.order_history[-limit:]
        if symbol:
            history = [o for o in history if o.symbol == symbol]
        return [o.to_dict() for o in history]
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        return {
            **self.execution_stats,
            'open_orders': len(self.orders),
            'fill_rate': self.execution_stats['filled_orders'] / self.execution_stats['total_orders'] if self.execution_stats['total_orders'] > 0 else 0
        }