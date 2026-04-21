"""
Order Executor for handling trades
"""
from typing import Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class OrderExecutor:
    """Execute orders on exchanges"""
    
    def __init__(self):
        self.orders = []
        self.active_orders = {}
        
    def execute_order(self, symbol: str, side: str, quantity: float, 
                      order_type: str = "MARKET", price: Optional[float] = None) -> Dict:
        """
        Execute an order
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            order_type: MARKET or LIMIT
            price: Limit price (required for LIMIT orders)
        
        Returns:
            Order execution result
        """
        order_id = f"ord_{datetime.now().timestamp()}"
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type,
            'status': 'PENDING',
            'timestamp': datetime.now().isoformat()
        }
        
        if order_type == "LIMIT" and price:
            order['price'] = price
            order['status'] = 'OPEN'
            self.active_orders[order_id] = order
        else:
            # Market order executes immediately
            order['status'] = 'FILLED'
            order['executed_price'] = price or 0
            order['executed_at'] = datetime.now().isoformat()
        
        self.orders.append(order)
        logger.info(f"Order executed: {side} {quantity} {symbol}")
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        if order_id in self.active_orders:
            self.active_orders[order_id]['status'] = 'CANCELLED'
            del self.active_orders[order_id]
            logger.info(f"Order {order_id} cancelled")
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        for order in self.orders:
            if order['id'] == order_id:
                return order
        return None
    
    def get_open_orders(self) -> Dict:
        """Get all open orders"""
        return self.active_orders
