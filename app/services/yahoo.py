"""
yahoo.py
Part of the app/services module.
Yahoo Finance integration for stock and crypto data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class YahooFinanceService:
    """
    Yahoo Finance service for fetching market data
    """
    
    def __init__(self):
        """Initialize Yahoo Finance service"""
        self.cache = {}
        self.cache_duration = 60  # seconds
        
    def get_klines(self, symbol: str, interval: str = "1h", 
                   period: str = "7d") -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Trading symbol
            interval: Time interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{interval}_{period}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_duration:
                return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval=interval, period=period)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Cache result
            self.cache[cache_key] = (df, datetime.now())
            
            logger.info(f"Fetched {len(df)} bars for {symbol} at {interval} interval")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for symbol
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Current price
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None
    
    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols
        
        Args:
            symbols: List of trading symbols
        
        Returns:
            Dictionary mapping symbols to quote data
        """
        results = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                results[symbol] = {
                    'symbol': symbol,
                    'price': info.get('regularMarketPrice', 0),
                    'change': info.get('regularMarketChange', 0),
                    'change_percent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('regularMarketVolume', 0),
                    'market_cap': info.get('marketCap', 0),
                    'timestamp': datetime.now()
                }
            except Exception as e:
                logger.error(f"Failed to get quote for {symbol}: {e}")
                results[symbol] = {'symbol': symbol, 'error': str(e)}
        
        return results
    
    def get_technical_indicators(self, symbol: str, 
                                 interval: str = "1h",
                                 period: str = "1mo") -> Dict:
        """
        Calculate technical indicators for a symbol
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            period: Data period
        
        Returns:
            Dictionary of latest indicator values
        """
        df = self.get_klines(symbol, interval, period)
        
        if df is None or df.empty:
            return {}
        
        close = df['close']
        
        # Calculate indicators
        indicators = {}
        
        # Moving averages
        indicators['SMA_20'] = close.rolling(20).mean().iloc[-1]
        indicators['SMA_50'] = close.rolling(50).mean().iloc[-1]
        indicators['EMA_12'] = close.ewm(span=12).mean().iloc[-1]
        indicators['EMA_26'] = close.ewm(span=26).mean().iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['RSI'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        indicators['MACD'] = macd.iloc[-1]
        indicators['MACD_Signal'] = signal.iloc[-1]
        indicators['MACD_Histogram'] = (macd - signal).iloc[-1]
        
        # Bollinger Bands
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        indicators['BB_Upper'] = (sma_20 + 2 * std_20).iloc[-1]
        indicators['BB_Lower'] = (sma_20 - 2 * std_20).iloc[-1]
        
        # ATR
        high = df['high']
        low = df['low']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        indicators['ATR'] = tr.rolling(14).mean().iloc[-1]
        
        # Volume indicators
        indicators['Volume'] = df['volume'].iloc[-1]
        indicators['Volume_SMA'] = df['volume'].rolling(20).mean().iloc[-1]
        
        # Volatility
        returns = close.pct_change().dropna()
        indicators['Volatility'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        return indicators
    
    def get_market_summary(self, symbol: str) -> Dict:
        """
        Get comprehensive market summary for a symbol
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Market summary dictionary
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            summary = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'current_price': info.get('regularMarketPrice', 0),
                'previous_close': info.get('regularMarketPreviousClose', 0),
                'day_high': info.get('regularMarketDayHigh', 0),
                'day_low': info.get('regularMarketDayLow', 0),
                'volume': info.get('regularMarketVolume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add technical indicators
            indicators = self.get_technical_indicators(symbol)
            summary['technical_indicators'] = indicators
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get market summary for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_crypto_data(self, symbol: str, vs_currency: str = 'usd',
                       days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get cryptocurrency data (uses yfinance crypto tickers)
        
        Args:
            symbol: Crypto symbol (e.g., BTC-USD)
            vs_currency: Quote currency
            days: Number of days of history
        
        Returns:
            DataFrame with price data
        """
        # Format symbol for yfinance (e.g., BTC-USD)
        ticker_symbol = f"{symbol.upper()}-{vs_currency.upper()}"
        return self.get_klines(ticker_symbol, interval="1h", period=f"{days}d")
    
    def get_forex_data(self, pair: str, interval: str = "1h",
                       period: str = "7d") -> Optional[pd.DataFrame]:
        """
        Get forex data
        
        Args:
            pair: Currency pair (e.g., EURUSD=X)
            interval: Time interval
            period: Data period
        
        Returns:
            DataFrame with OHLCV data
        """
        # Format pair for yfinance
        ticker_symbol = f"{pair}=X"
        return self.get_klines(ticker_symbol, interval, period)
    
    def clear_cache(self):
        """Clear cached data"""
        self.cache.clear()
        logger.info("Yahoo Finance cache cleared")
    
    def place_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """
        Place order (mock - Yahoo Finance doesn't support trading)
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
        
        Returns:
            Mock order response
        """
        logger.warning("Yahoo Finance does not support trading. This is a mock.")
        
        current_price = self.get_current_price(symbol)
        
        return {
            'success': True,
            'id': f"mock_{datetime.now().timestamp()}",
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': current_price or 0,
            'status': 'FILLED',
            'timestamp': datetime.now().isoformat()
        }
# Add alias for backward compatibility
YahooService = YahooFinanceService
