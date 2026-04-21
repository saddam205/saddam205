"""
trade.py
Part of the app/schemas module.
Pydantic schemas for trading operations.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderStatus(str, Enum):
    """Order status"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TradeMode(str, Enum):
    """Trading mode"""
    REAL = "REAL"
    VIRTUAL = "VIRTUAL"
    PAPER = "PAPER"


class TradeSignal(BaseModel):
    """Trading signal schema"""
    signal: str = Field(..., description="Trading signal: BUY, SELL, HOLD")
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence (0-1)")
    timestamp: datetime = Field(default_factory=datetime.now)
    symbol: str = Field(..., description="Trading symbol")
    price: Optional[float] = Field(None, description="Current price")
    reasoning: Optional[str] = Field(None, description="Signal reasoning")
    
    @validator('signal')
    def validate_signal(cls, v):
        if v.upper() not in ['BUY', 'SELL', 'HOLD']:
            raise ValueError('Signal must be BUY, SELL, or HOLD')
        return v.upper()


class TradeRequest(BaseModel):
    """Trade execution request"""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    side: OrderSide = Field(..., description="BUY or SELL")
    quantity: float = Field(..., gt=0, description="Order quantity")
    order_type: OrderType = Field(OrderType.MARKET, description="Order type")
    price: Optional[float] = Field(None, description="Limit price (required for LIMIT orders)")
    stop_price: Optional[float] = Field(None, description="Stop price for stop orders")
    mode: TradeMode = Field(TradeMode.VIRTUAL, description="Trading mode")
    confidence: float = Field(0.7, ge=0, le=1, description="Signal confidence")
    client_order_id: Optional[str] = Field(None, description="Client-side order ID")
    
    @validator('price')
    def validate_price(cls, v, values):
        if values.get('order_type') == OrderType.LIMIT and v is None:
            raise ValueError('Price is required for LIMIT orders')
        if v is not None and v <= 0:
            raise ValueError('Price must be positive')
        return v
    
    @validator('stop_price')
    def validate_stop_price(cls, v, values):
        if values.get('order_type') in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT] and v is None:
            raise ValueError('Stop price is required for stop orders')
        return v


class OrderResponse(BaseModel):
    """Order execution response"""
    order_id: str = Field(..., description="Order ID")
    client_order_id: Optional[str] = Field(None, description="Client order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side")
    order_type: OrderType = Field(..., description="Order type")
    quantity: float = Field(..., description="Order quantity")
    filled_quantity: float = Field(0, description="Filled quantity")
    price: Optional[float] = Field(None, description="Limit price")
    avg_fill_price: float = Field(0, description="Average fill price")
    status: OrderStatus = Field(..., description="Order status")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    mode: TradeMode = Field(..., description="Trading mode")
    
    @property
    def remaining_quantity(self) -> float:
        """Calculate remaining quantity"""
        return self.quantity - self.filled_quantity


class TradeResponse(BaseModel):
    """Trade execution response (legacy compatibility)"""
    success: bool = Field(..., description="Whether trade was successful")
    message: str = Field(..., description="Response message")
    trade_id: Optional[str] = Field(None, description="Trade ID")
    signal: Optional[str] = Field(None, description="Trading signal")
    confidence: Optional[float] = Field(None, description="Signal confidence")
    position_size: Optional[float] = Field(None, description="Position size")
    price: Optional[float] = Field(None, description="Execution price")
    timestamp: datetime = Field(default_factory=datetime.now)
    order_details: Optional[OrderResponse] = Field(None, description="Detailed order info")


class Position(BaseModel):
    """Position schema"""
    symbol: str = Field(..., description="Asset symbol")
    quantity: float = Field(..., description="Position quantity")
    entry_price: float = Field(..., description="Average entry price")
    current_price: float = Field(..., description="Current market price")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")
    unrealized_pnl_percent: float = Field(..., description="Unrealized P&L percentage")
    side: str = Field(..., description="LONG or SHORT")
    entry_time: datetime = Field(..., description="Position entry time")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    
    @validator('side')
    def validate_side(cls, v):
        if v.upper() not in ['LONG', 'SHORT']:
            raise ValueError('Side must be LONG or SHORT')
        return v.upper()


class PositionResponse(BaseModel):
    """Position list response"""
    positions: List[Position] = Field(default_factory=list)
    total_positions: int = Field(0)
    total_exposure: float = Field(0, description="Total position value")
    total_unrealized_pnl: float = Field(0)


class ClosePositionRequest(BaseModel):
    """Request to close a position"""
    symbol: str = Field(..., description="Symbol to close")
    quantity: Optional[float] = Field(None, description="Quantity to close (default: all)")
    reduce_only: bool = Field(True, description="Reduce only mode")


class UpdatePositionRequest(BaseModel):
    """Request to update position parameters"""
    symbol: str = Field(..., description="Position symbol")
    stop_loss: Optional[float] = Field(None, description="New stop loss price")
    take_profit: Optional[float] = Field(None, description="New take profit price")
    
    @validator('stop_loss', 'take_profit')
    def validate_price(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Price must be positive')
        return v


class TradeHistoryRequest(BaseModel):
    """Trade history query request"""
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    start_date: Optional[datetime] = Field(None, description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date")
    limit: int = Field(100, ge=1, le=1000, description="Results limit")
    offset: int = Field(0, ge=0, description="Results offset")


class CancelOrderRequest(BaseModel):
    """Cancel order request"""
    order_id: str = Field(..., description="Order ID to cancel")
    symbol: Optional[str] = Field(None, description="Symbol (for exchange lookup)")


class ModifyOrderRequest(BaseModel):
    """Modify existing order"""
    order_id: str = Field(..., description="Order ID to modify")
    quantity: Optional[float] = Field(None, gt=0, description="New quantity")
    price: Optional[float] = Field(None, gt=0, description="New limit price")
    stop_price: Optional[float] = Field(None, gt=0, description="New stop price")