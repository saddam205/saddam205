"""
exceptions.py
Part of the app/utils module.
Custom exceptions for trading system.
"""


class TradingException(Exception):
    """Base exception for trading system"""
    def __init__(self, message: str, code: str = "TRADING_ERROR", details: dict = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class OrderException(TradingException):
    """Exception for order-related errors"""
    def __init__(self, message: str, order_id: str = None, details: dict = None):
        super().__init__(message, code="ORDER_ERROR", details=details)
        self.order_id = order_id


class RiskException(TradingException):
    """Exception for risk management violations"""
    def __init__(self, message: str, risk_limit: str = None, details: dict = None):
        super().__init__(message, code="RISK_ERROR", details=details)
        self.risk_limit = risk_limit


class ConfigurationException(TradingException):
    """Exception for configuration errors"""
    def __init__(self, message: str, config_key: str = None, details: dict = None):
        super().__init__(message, code="CONFIG_ERROR", details=details)
        self.config_key = config_key


class DataException(TradingException):
    """Exception for data-related errors"""
    def __init__(self, message: str, data_source: str = None, details: dict = None):
        super().__init__(message, code="DATA_ERROR", details=details)
        self.data_source = data_source


class APIException(TradingException):
    """Exception for API-related errors"""
    def __init__(self, message: str, endpoint: str = None, status_code: int = None, details: dict = None):
        super().__init__(message, code="API_ERROR", details=details)
        self.endpoint = endpoint
        self.status_code = status_code


class ValidationException(TradingException):
    """Exception for validation errors"""
    def __init__(self, message: str, field: str = None, value: any = None, details: dict = None):
        super().__init__(message, code="VALIDATION_ERROR", details=details)
        self.field = field
        self.value = value


class InsufficientBalanceException(OrderException):
    """Exception for insufficient balance"""
    def __init__(self, message: str, required: float = None, available: float = None):
        super().__init__(message, details={'required': required, 'available': available})
        self.required = required
        self.available = available


class PositionNotFoundException(OrderException):
    """Exception when position not found"""
    def __init__(self, symbol: str):
        super().__init__(f"Position not found for symbol: {symbol}", details={'symbol': symbol})
        self.symbol = symbol


class KillSwitchActivatedException(TradingException):
    """Exception when kill switch is active"""
    def __init__(self, reason: str = None):
        super().__init__(
            f"Trading halted: Kill switch active" + (f" ({reason})" if reason else ""),
            code="KILL_SWITCH_ACTIVE"
        )
        self.reason = reason


class RateLimitExceededException(APIException):
    """Exception for rate limit exceeded"""
    def __init__(self, retry_after: int = None):
        super().__init__(
            "Rate limit exceeded",
            status_code=429,
            details={'retry_after': retry_after}
        )
        self.retry_after = retry_after


class ModelNotTrainedException(TradingException):
    """Exception when model is not trained"""
    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' has not been trained yet",
            code="MODEL_NOT_TRAINED"
        )
        self.model_name = model_name


class InvalidSignalException(TradingException):
    """Exception for invalid trading signals"""
    def __init__(self, signal: str, reason: str = None):
        super().__init__(
            f"Invalid signal: {signal}" + (f" ({reason})" if reason else ""),
            code="INVALID_SIGNAL"
        )
        self.signal = signal