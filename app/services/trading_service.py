import asyncio
import uuid
from datetime import datetime
from typing import Dict, List
import logging
import json

from app.services.binance import BinanceService
from app.models.position_sizer import DynamicPositionSizer
from app.config import config

logger = logging.getLogger(__name__)

class TradingService:
    def __init__(self):
        self.binance = BinanceService()
        self.position_sizer = DynamicPositionSizer()
        self.open_positions = {}
        self.trade_history = []
        
    async def calculate_signal(self, df, selected_indicators):
        """
        Calculate trading signal using AI-selected indicators
        This is where your trained XGBoost model would be used
        """
        # Placeholder for actual AI model prediction
        # In production, load your trained model and predict
        
        # Simulate signal based on indicators
        last_row = df.iloc[-1]
        
        # Simple simulation for demo
        signal_score = 0
        if 'RSI_14' in selected_indicators:
            rsi = last_row.get('RSI_14', 50)
            if rsi < 30:
                signal_score += 1
            elif rsi > 70:
                signal_score -= 1
        
        if 'MACD' in selected_indicators:
            macd = last_row.get('MACD', 0)
            macd_signal = last_row.get('MACD_Signal', 0)
            if macd > macd_signal:
                signal_score += 1
            else:
                signal_score -= 1
        
        if 'Volume_Ratio' in selected_indicators:
            vol_ratio = last_row.get('Volume_Ratio', 1)
            if vol_ratio > 1.2:
                signal_score += 0.5
        
        # Determine signal
        if signal_score > 1:
            signal = "BUY"
            confidence = min(0.5 + (signal_score / 10), 0.85)
        elif signal_score < -1:
            signal = "SELL"
            confidence = min(0.5 + (abs(signal_score) / 10), 0.85)
        else:
            signal = "HOLD"
            confidence = 0.5
        
        # Get volatility
        volatility = last_row.get('ATR_Pct', 0.02)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'volatility': volatility,
            'signal_score': signal_score,
            'timestamp': datetime.now()
        }
    
    async def execute_real_trade(self, symbol, side, quantity, confidence):
        """Execute real trade on Binance"""
        try:
            # Place order
            order = self.binance.place_market_order(symbol, side, quantity)
            
            if order:
                trade_id = str(uuid.uuid4())
                trade = {
                    'id': trade_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': float(order['fills'][0]['price']),
                    'confidence': confidence,
                    'timestamp': datetime.now(),
                    'status': 'OPEN'
                }
                
                # Set stop loss and take profit
                await self.set_trade_protections(trade)
                
                self.open_positions[trade_id] = trade
                self.trade_history.append(trade)
                
                logger.info(f"Real trade executed: {side} {quantity} {symbol}")
                return trade
            
        except Exception as e:
            logger.error(f"Real trade failed: {e}")
            return None
    
    async def set_trade_protections(self, trade):
        """Set stop loss and take profit for a trade"""
        entry_price = trade['price']
        
        trade['stop_loss'] = entry_price * (1 - config.STOP_LOSS)
        trade['take_profit'] = entry_price * (1 + config.TAKE_PROFIT)
        
        # In production, you would place limit orders here
        logger.info(f"Protections set: SL={trade['stop_loss']:.2f}, TP={trade['take_profit']:.2f}")
    
    async def log_trade(self, trade, signal_data, indicators, sizing_info):
        """Log trade details to file"""
        log_entry = {
            'trade': trade,
            'signal': signal_data,
            'indicators_used': indicators,
            'sizing': sizing_info,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"data/trades/trade_{trade['id']}.json"
        with open(filename, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)
        
        logger.info(f"Trade logged: {filename}")
    
    def get_open_positions(self):
        """Get all open positions"""
        return list(self.open_positions.values())
    
    def get_performance(self):
        """Calculate performance metrics"""
        if not self.trade_history:
            return {"message": "No trades yet"}
        
        closed_trades = [t for t in self.trade_history if t['status'] == 'CLOSED']
        
        if not closed_trades:
            return {"message": "No closed trades"}
        
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(closed_trades) if closed_trades else 0
        }