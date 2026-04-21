import uuid
from datetime import datetime
from typing import Dict, List
import json
import os

class VirtualService:
    def __init__(self):
        self.virtual_balance = 500000  # $500,000 test account
        self.virtual_positions = {}
        self.virtual_trades = []
        self.initial_balance = 500000
        
    async def execute_virtual_trade(self, symbol, side, quantity, price, confidence):
        """Execute virtual/paper trade"""
        trade_id = str(uuid.uuid4())
        
        if side == "BUY":
            cost = quantity * price
            if cost > self.virtual_balance:
                return {
                    'success': False,
                    'message': f'Insufficient balance. Need ${cost:,.2f}, have ${self.virtual_balance:,.2f}'
                }
            
            self.virtual_balance -= cost
            
            trade = {
                'id': trade_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': price,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'status': 'OPEN',
                'stop_loss': price * 0.98,
                'take_profit': price * 1.04
            }
            
            self.virtual_positions[trade_id] = trade
            self.virtual_trades.append(trade)
            
            return trade
            
        elif side == "SELL":
            # Find open buy position
            open_position = next((p for p in self.virtual_positions.values() if p['symbol'] == symbol), None)
            
            if not open_position:
                return {'success': False, 'message': 'No open position to sell'}
            
            revenue = quantity * price
            self.virtual_balance += revenue
            
            pnl = revenue - (open_position['quantity'] * open_position['entry_price'])
            pnl_pct = (price / open_position['entry_price'] - 1) * 100
            
            trade = {
                'id': open_position['id'],
                'symbol': symbol,
                'side': 'SELL',
                'quantity': quantity,
                'entry_price': open_position['entry_price'],
                'exit_price': price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'confidence': open_position['confidence'],
                'timestamp': datetime.now(),
                'status': 'CLOSED'
            }
            
            # Remove position and add to history
            del self.virtual_positions[open_position['id']]
            self.virtual_trades.append(trade)
            
            return trade
        
        return None
    
    def get_virtual_balance(self):
        """Get current virtual balance"""
        return self.virtual_balance
    
    def get_performance(self):
        """Get virtual trading performance"""
        closed_trades = [t for t in self.virtual_trades if t.get('status') == 'CLOSED']
        
        if not closed_trades:
            return {
                'balance': self.virtual_balance,
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0
            }
        
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        total_return = ((self.virtual_balance - self.initial_balance) / self.initial_balance) * 100
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.virtual_balance,
            'total_return': total_return,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(closed_trades) - len(winning_trades),
            'win_rate': (len(winning_trades) / len(closed_trades)) * 100,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(closed_trades),
            'open_positions': len(self.virtual_positions)
        }
    
    def save_virtual_trades(self):
        """Save virtual trades to file"""
        trades_file = "data/trades/virtual_trades.json"
        with open(trades_file, 'w') as f:
            json.dump({
                'initial_balance': self.initial_balance,
                'current_balance': self.virtual_balance,
                'trades': self.virtual_trades,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)