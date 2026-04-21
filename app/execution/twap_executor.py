"""
twap_executor.py
Part of the app/execution module.
Time-Weighted Average Price (TWAP) algorithm for large order execution.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import numpy as np

from .order_executor import OrderExecutor, OrderSide, OrderType

logger = logging.getLogger(__name__)


@dataclass
class TWAPOrder:
    """TWAP order configuration"""
    symbol: str
    side: OrderSide
    total_quantity: float
    duration_minutes: int
    slices: int = 10
    start_time: Optional[datetime] = None
    aggressive: bool = False
    min_slice_size: float = 0.01
    execution_callback: Optional[Callable] = None
    status: str = "PENDING"
    executed_quantity: float = 0.0
    avg_price: float = 0.0
    slices_executed: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def slice_size(self) -> float:
        """Calculate slice size"""
        return self.total_quantity / self.slices
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity"""
        return self.total_quantity - self.executed_quantity
    
    @property
    def is_complete(self) -> bool:
        """Check if TWAP order is complete"""
        return self.executed_quantity >= self.total_quantity
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'total_quantity': self.total_quantity,
            'executed_quantity': self.executed_quantity,
            'remaining_quantity': self.remaining_quantity,
            'duration_minutes': self.duration_minutes,
            'slices': self.slices,
            'slices_executed': self.slices_executed,
            'avg_price': self.avg_price,
            'status': self.status,
            'completion_pct': self.executed_quantity / self.total_quantity * 100 if self.total_quantity > 0 else 0
        }


class TWAPExecutor:
    """
    Time-Weighted Average Price (TWAP) executor
    Splits large orders into smaller slices to minimize market impact
    """
    
    def __init__(self, order_executor: OrderExecutor, check_interval: int = 5):
        """
        Initialize TWAP executor
        
        Args:
            order_executor: Underlying order executor
            check_interval: Interval between slice checks (seconds)
        """
        self.executor = order_executor
        self.check_interval = check_interval
        self.active_twap_orders: Dict[str, TWAPOrder] = {}
        self.completed_twap_orders: List[TWAPOrder] = []
        
    async def execute_twap(self, symbol: str, side: str, total_quantity: float,
                          duration_minutes: int, slices: int = 10,
                          aggressive: bool = False,
                          callback: Optional[Callable] = None) -> Dict:
        """
        Execute a TWAP order
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            total_quantity: Total quantity to execute
            duration_minutes: Duration over which to execute
            slices: Number of slices to split into
            aggressive: Execute faster if price moves favorably
            callback: Callback for slice execution
        
        Returns:
            TWAP order details
        """
        order_id = f"twap_{symbol}_{datetime.now().timestamp()}"
        
        twap_order = TWAPOrder(
            symbol=symbol,
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            total_quantity=total_quantity,
            duration_minutes=duration_minutes,
            slices=slices,
            aggressive=aggressive,
            execution_callback=callback,
            start_time=datetime.now() + timedelta(seconds=5)  # Start in 5 seconds
        )
        
        self.active_twap_orders[order_id] = twap_order
        
        # Start execution task
        asyncio.create_task(self._execute_slices(order_id))
        
        logger.info(f"Started TWAP order {order_id}: {total_quantity} {symbol} over {duration_minutes} minutes")
        
        return {
            'order_id': order_id,
            **twap_order.to_dict()
        }
    
    async def _execute_slices(self, order_id: str):
        """Execute TWAP slices"""
        twap = self.active_twap_orders.get(order_id)
        if not twap:
            return
        
        # Calculate slice timing
        slice_interval = (twap.duration_minutes * 60) / twap.slices
        slice_interval = max(slice_interval, self.check_interval)
        
        # Wait for start time
        if twap.start_time:
            wait_seconds = (twap.start_time - datetime.now()).total_seconds()
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
        
        try:
            while not twap.is_complete:
                # Calculate slice size for this iteration
                slice_size = self._calculate_slice_size(twap)
                
                if slice_size < twap.min_slice_size:
                    logger.debug(f"Slice too small ({slice_size}), skipping")
                    await asyncio.sleep(slice_interval)
                    continue
                
                # Execute slice
                result = await self._execute_slice(twap, slice_size)
                
                if result.get('success'):
                    twap.slices_executed += 1
                    twap.executed_quantity += result.get('filled_quantity', slice_size)
                    
                    # Update average price
                    fill_price = result.get('avg_price', 0)
                    total_cost = twap.avg_price * (twap.executed_quantity - result.get('filled_quantity'))
                    total_cost += fill_price * result.get('filled_quantity')
                    twap.avg_price = total_cost / twap.executed_quantity if twap.executed_quantity > 0 else 0
                    
                    # Callback
                    if twap.execution_callback:
                        await twap.execution_callback(result)
                    
                    logger.info(f"TWAP slice executed: {result.get('filled_quantity')} {twap.symbol} @ {fill_price:.2f}")
                else:
                    logger.error(f"TWAP slice failed: {result.get('error')}")
                
                # Check if we should adjust schedule
                if twap.aggressive and self._should_accelerate(twap):
                    slice_interval = max(slice_interval * 0.7, self.check_interval)
                
                # Wait for next slice
                await asyncio.sleep(slice_interval)
            
            # Order complete
            twap.status = "COMPLETED"
            self.completed_twap_orders.append(twap)
            del self.active_twap_orders[order_id]
            
            logger.info(f"TWAP order {order_id} completed: {twap.executed_quantity}/{twap.total_quantity} @ {twap.avg_price:.2f}")
            
        except Exception as e:
            logger.error(f"TWAP execution failed: {e}")
            twap.status = "FAILED"
            del self.active_twap_orders[order_id]
    
    def _calculate_slice_size(self, twap: TWAPOrder) -> float:
        """Calculate slice size for current iteration"""
        # Base slice size
        base_slice = twap.slice_size
        
        # Adjust for remaining time
        elapsed = (datetime.now() - twap.created_at).total_seconds() / 60
        remaining_time = max(0.1, twap.duration_minutes - elapsed)
        remaining_quantity = twap.remaining_quantity
        
        # Time-weighted adjustment
        if remaining_time > 0:
            # Execute faster if behind schedule
            expected_progress = elapsed / twap.duration_minutes
            actual_progress = twap.executed_quantity / twap.total_quantity
            
            if actual_progress < expected_progress:
                # Catch up: increase slice size
                catchup_factor = min(2.0, (expected_progress - actual_progress) * 5 + 1)
                base_slice *= catchup_factor
        
        # Aggressive mode: increase slice size on favorable moves
        if twap.aggressive:
            # This would check price movement
            base_slice *= 1.2
        
        return min(base_slice, remaining_quantity)
    
    def _should_accelerate(self, twap: TWAPOrder) -> bool:
        """Check if execution should be accelerated"""
        # Placeholder for price-based acceleration logic
        return False
    
    async def _execute_slice(self, twap: TWAPOrder, slice_size: float) -> Dict:
        """Execute a single slice"""
        return await self.executor.execute(
            symbol=twap.symbol,
            side=twap.side.value,
            quantity=slice_size,
            order_type="MARKET"
        )
    
    def cancel_twap(self, order_id: str) -> Dict:
        """Cancel a TWAP order"""
        if order_id in self.active_twap_orders:
            twap = self.active_twap_orders[order_id]
            twap.status = "CANCELLED"
            del self.active_twap_orders[order_id]
            
            logger.info(f"TWAP order {order_id} cancelled")
            
            return {
                'success': True,
                'order_id': order_id,
                'executed_quantity': twap.executed_quantity,
                'remaining_quantity': twap.remaining_quantity
            }
        
        return {'success': False, 'error': 'TWAP order not found'}
    
    def get_active_twaps(self) -> List[Dict]:
        """Get active TWAP orders"""
        return [
            {'order_id': oid, **twap.to_dict()}
            for oid, twap in self.active_twap_orders.items()
        ]
    
    def get_twap_history(self, limit: int = 50) -> List[Dict]:
        """Get completed TWAP orders"""
        return [twap.to_dict() for twap in self.completed_twap_orders[-limit:]]