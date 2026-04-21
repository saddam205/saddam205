import ccxt
import pandas as pd
from typing import Dict, List, Optional
from app.config import settings

class BinanceService:
    def __init__(self):
        self.client = ccxt.binance({
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_API_SECRET,
            'enableRateLimit': True,
        })

    def get_balance(self):
        return self.client.fetch_balance()

# Ensure other files can import it as BinanceClient if they need to
BinanceClient = BinanceService 
