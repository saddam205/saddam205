"""
validators.py
Part of the app/utils module.
Input validation utilities for trading operations.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal, InvalidOperation
import logging

from .exceptions import ValidationException

logger = logging.getLogger(__name__)


def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format
    
    Args:
        symbol: Trading symbol (e.g., BTCUSDT, ETH-USD)
    
    Returns:
        Whether symbol is valid
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Common patterns for trading symbols
    patterns = [
        r'^[A-Z]{2,10}$',  # Simple symbol
        r'^[A-Z]{2,6}[A-Z]{3,5}$',  # Base+Quote (BTCUSDT)
        r'^[A-Z]{2,6}-[A-Z]{3,5}$',  # Dash separated (BTC-USD)
        r'^[A-Z]{2,6}/[A-Z]{3,5}$',  # Slash separated (BTC/USD)
    ]
    
    for pattern in patterns:
        if re.match(pattern, symbol.upper()):
            return True
    
    return False


def validate_quantity(quantity: float, min_quantity: float = 0.00001, 
                      max_quantity: float = 10000) -> Tuple[bool, str]:
    """
    Validate order quantity
    
    Args:
        quantity: Order quantity
        min_quantity: Minimum allowed quantity
        max_quantity: Maximum allowed quantity
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(quantity, (int, float)):
        return False, f"Quantity must be a number, got {type(quantity)}"
    
    if quantity <= 0:
        return False, f"Quantity must be positive, got {quantity}"
    
    if quantity < min_quantity:
        return False, f"Quantity {quantity} below minimum {min_quantity}"
    
    if quantity > max_quantity:
        return False, f"Quantity {quantity} exceeds maximum {max_quantity}"
    
    return True, ""


def validate_price(price: float, min_price: float = 0.00000001, 
                   max_price: float = 1_000_000) -> Tuple[bool, str]:
    """
    Validate order price
    
    Args:
        price: Order price
        min_price: Minimum allowed price
        max_price: Maximum allowed price
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(price, (int, float)):
        return False, f"Price must be a number, got {type(price)}"
    
    if price <= 0:
        return False, f"Price must be positive, got {price}"
    
    if price < min_price:
        return False, f"Price {price} below minimum {min_price}"
    
    if price > max_price:
        return False, f"Price {price} exceeds maximum {max_price}"
    
    return True, ""


def validate_order_params(symbol: str, side: str, quantity: float, 
                          price: float = None, order_type: str = "MARKET") -> Dict:
    """
    Validate order parameters
    
    Args:
        symbol: Trading symbol
        side: BUY or SELL
        quantity: Order quantity
        price: Order price (required for LIMIT orders)
        order_type: Order type
    
    Returns:
        Dictionary with validation result
    
    Raises:
        ValidationException: If validation fails
    """
    errors = []
    
    # Validate symbol
    if not validate_symbol(symbol):
        errors.append(f"Invalid symbol: {symbol}")
    
    # Validate side
    if side.upper() not in ['BUY', 'SELL']:
        errors.append(f"Invalid side: {side}. Must be BUY or SELL")
    
    # Validate quantity
    is_valid, error = validate_quantity(quantity)
    if not is_valid:
        errors.append(error)
    
    # Validate price for limit orders
    if order_type.upper() == 'LIMIT':
        if price is None:
            errors.append("Price is required for LIMIT orders")
        else:
            is_valid, error = validate_price(price)
            if not is_valid:
                errors.append(error)
    
    # Validate order type
    valid_types = ['MARKET', 'LIMIT', 'STOP_LOSS', 'STOP_LIMIT', 'TRAILING_STOP']
    if order_type.upper() not in valid_types:
        errors.append(f"Invalid order type: {order_type}")
    
    if errors:
        raise ValidationException(
            "Order validation failed",
            field="order_params",
            value={'symbol': symbol, 'side': side, 'quantity': quantity},
            details={'errors': errors}
        )
    
    return {
        'valid': True,
        'symbol': symbol.upper(),
        'side': side.upper(),
        'quantity': quantity,
        'price': price,
        'order_type': order_type.upper()
    }


class DataValidator:
    """
    Validates market data and DataFrame structures
    """
    
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    @classmethod
    def validate_ohlcv(cls, df: 'pd.DataFrame') -> Tuple[bool, str]:
        """
        Validate OHLCV DataFrame
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        import pandas as pd
        
        if df is None:
            return False, "DataFrame is None"
        
        if df.empty:
            return False, "DataFrame is empty"
        
        # Check required columns
        missing = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            return False, f"Missing columns: {missing}"
        
        # Check for NaN values
        nan_cols = df[cls.REQUIRED_COLUMNS].isna().any()
        if nan_cols.any():
            return False, f"NaN values found in columns: {nan_cols[nan_cols].index.tolist()}"
        
        # Check for non-positive prices
        for col in ['open', 'high', 'low', 'close']:
            if (df[col] <= 0).any():
                return False, f"Non-positive values in {col}"
        
        # Check high >= low
        if (df['high'] < df['low']).any():
            return False, "High is less than low in some rows"
        
        # Check close within high-low range
        if ((df['close'] > df['high']) | (df['close'] < df['low'])).any():
            return False, "Close outside high-low range"
        
        return True, ""
    
    @classmethod
    def validate_indicators(cls, df: 'pd.DataFrame', 
                           indicator_columns: List[str]) -> Tuple[bool, str]:
        """
        Validate indicator columns
        
        Args:
            df: DataFrame to validate
            indicator_columns: List of indicator column names
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        missing = [col for col in indicator_columns if col not in df.columns]
        if missing:
            return False, f"Missing indicator columns: {missing}"
        
        # Check for excessive NaN values (more than 10%)
        for col in indicator_columns:
            nan_pct = df[col].isna().mean()
            if nan_pct > 0.1:
                return False, f"Column {col} has {nan_pct:.1%} NaN values"
        
        return True, ""


class InputValidator:
    """
    Validates user inputs and configuration parameters
    """
    
    @staticmethod
    def validate_percentage(value: float, name: str, 
                           min_val: float = 0, max_val: float = 100) -> float:
        """
        Validate percentage value
        
        Args:
            value: Percentage value
            name: Parameter name (for error message)
            min_val: Minimum allowed value
            max_val: Maximum allowed value
        
        Returns:
            Validated value
        
        Raises:
            ValidationException: If validation fails
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValidationException(
                f"{name} must be a number",
                field=name,
                value=value
            )
        
        if value < min_val or value > max_val:
            raise ValidationException(
                f"{name} must be between {min_val} and {max_val}",
                field=name,
                value=value
            )
        
        return value
    
    @staticmethod
    def validate_positive_number(value: float, name: str, 
                                 allow_zero: bool = False) -> float:
        """
        Validate positive number
        
        Args:
            value: Number to validate
            name: Parameter name
            allow_zero: Whether zero is allowed
        
        Returns:
            Validated value
        
        Raises:
            ValidationException: If validation fails
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValidationException(
                f"{name} must be a number",
                field=name,
                value=value
            )
        
        if allow_zero:
            if value < 0:
                raise ValidationException(
                    f"{name} must be non-negative",
                    field=name,
                    value=value
                )
        else:
            if value <= 0:
                raise ValidationException(
                    f"{name} must be positive",
                    field=name,
                    value=value
                )
        
        return value
    
    @staticmethod
    def validate_string(value: str, name: str, 
                        min_length: int = 1, max_length: int = 100) -> str:
        """
        Validate string value
        
        Args:
            value: String to validate
            name: Parameter name
            min_length: Minimum length
            max_length: Maximum length
        
        Returns:
            Validated string
        
        Raises:
            ValidationException: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationException(
                f"{name} must be a string",
                field=name,
                value=value
            )
        
        if len(value) < min_length:
            raise ValidationException(
                f"{name} must be at least {min_length} characters",
                field=name,
                value=value
            )
        
        if len(value) > max_length:
            raise ValidationException(
                f"{name} must be at most {max_length} characters",
                field=name,
                value=value
            )
        
        return value.strip()
    
    @staticmethod
    def validate_choice(value: str, choices: List[str], name: str) -> str:
        """
        Validate that value is in allowed choices
        
        Args:
            value: Value to validate
            choices: List of allowed choices
            name: Parameter name
        
        Returns:
            Validated value
        
        Raises:
            ValidationException: If validation fails
        """
        if value not in choices:
            raise ValidationException(
                f"{name} must be one of: {', '.join(choices)}",
                field=name,
                value=value
            )
        
        return value
    
    @staticmethod
    def validate_decimal(value: str, name: str) -> Decimal:
        """
        Validate decimal value
        
        Args:
            value: String to validate as decimal
            name: Parameter name
        
        Returns:
            Decimal value
        
        Raises:
            ValidationException: If validation fails
        """
        try:
            return Decimal(str(value))
        except InvalidOperation:
            raise ValidationException(
                f"{name} must be a valid decimal number",
                field=name,
                value=value
            )
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> str:
        """
        Validate timeframe string
        
        Args:
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
        
        Returns:
            Validated timeframe
        
        Raises:
            ValidationException: If validation fails
        """
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M']
        
        if timeframe not in valid_timeframes:
            raise ValidationException(
                f"Invalid timeframe: {timeframe}. Valid: {', '.join(valid_timeframes)}",
                field='timeframe',
                value=timeframe
            )
        
        return timeframe