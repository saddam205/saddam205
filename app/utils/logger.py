"""
logger.py
Part of the app/utils module.
Structured logging for trading system with rotation and monitoring.
"""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_entry['extra'] = record.extra_data
        
        return json.dumps(log_entry)


class TradingLogger:
    """
    Centralized logging system for trading operations
    """
    
    def __init__(self, log_dir: str = "data/logs", 
                 log_level: str = "INFO",
                 json_format: bool = True):
        """
        Initialize trading logger
        
        Args:
            log_dir: Directory for log files
            log_level: Logging level
            json_format: Use JSON format for logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('trading')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation (10MB per file, keep 10 backups)
        file_handler = RotatingFileHandler(
            self.log_dir / 'trading.log',
            maxBytes=10_000_000,
            backupCount=10
        )
        
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(console_format)
        
        self.logger.addHandler(file_handler)
        
        # Daily rotation for separate logs
        daily_handler = TimedRotatingFileHandler(
            self.log_dir / 'trading_daily.log',
            when='midnight',
            interval=1,
            backupCount=30
        )
        daily_handler.setFormatter(JSONFormatter() if json_format else console_format)
        self.logger.addHandler(daily_handler)
        
        # Error-specific handler
        error_handler = RotatingFileHandler(
            self.log_dir / 'errors.log',
            maxBytes=10_000_000,
            backupCount=20
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter() if json_format else console_format)
        self.logger.addHandler(error_handler)
        
    def info(self, message: str, extra: Dict[str, Any] = None):
        """Log info message"""
        self._log(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Dict[str, Any] = None):
        """Log warning message"""
        self._log(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Dict[str, Any] = None):
        """Log error message"""
        self._log(logging.ERROR, message, extra)
    
    def debug(self, message: str, extra: Dict[str, Any] = None):
        """Log debug message"""
        self._log(logging.DEBUG, message, extra)
    
    def critical(self, message: str, extra: Dict[str, Any] = None):
        """Log critical message"""
        self._log(logging.CRITICAL, message, extra)
    
    def _log(self, level: int, message: str, extra: Dict[str, Any] = None):
        """Internal logging method"""
        if extra:
            # Create a log record with extra data
            self.logger.log(level, message, extra={'extra_data': extra})
        else:
            self.logger.log(level, message)
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade execution"""
        self.info(
            f"Trade executed: {trade_data.get('side')} {trade_data.get('quantity')} {trade_data.get('symbol')}",
            extra=trade_data
        )
    
    def log_signal(self, signal_data: Dict[str, Any]):
        """Log trading signal"""
        self.info(
            f"Signal: {signal_data.get('signal')} (conf={signal_data.get('confidence', 0):.2%})",
            extra=signal_data
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        self.error(
            str(error),
            extra={'error_type': type(error).__name__, 'context': context}
        )
    
    def get_logger(self) -> logging.Logger:
        """Get underlying logger instance"""
        return self.logger


# Global logger instance
_default_logger: Optional[TradingLogger] = None


def setup_logger(log_dir: str = "data/logs", 
                 log_level: str = "INFO",
                 json_format: bool = True) -> TradingLogger:
    """
    Setup global trading logger
    
    Args:
        log_dir: Log directory path
        log_level: Logging level
        json_format: Use JSON format
    
    Returns:
        TradingLogger instance
    """
    global _default_logger
    _default_logger = TradingLogger(log_dir, log_level, json_format)
    return _default_logger


def get_logger() -> TradingLogger:
    """
    Get global trading logger
    
    Returns:
        TradingLogger instance
    """
    if _default_logger is None:
        return setup_logger()
    return _default_logger