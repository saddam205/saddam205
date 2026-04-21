import joblib
import pandas as pd
import numpy as np
from binance.client import Client
import time
from datetime import datetime
import json

class VirtualTradingBot:
    def __init__(self, initial_balance=500000):
        """Initialize virtual trading system with $500,000 test account"""
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.trades = []
        self.daily_pnl = []
        self.current_day = None
        
        # Load model
        self.model = joblib.load("advanced_trading_model.pkl")
        
        # Load features
        with open("advanced_features.txt", "r") as f:
            self.features = [line.strip() for line in f.readlines()]
        
        # Binance client (for data only)
        self.client = Client()
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'peak_balance': initial_balance,
            'sharpe_ratio': 0
        }
        
        print("="*60)
        print(f"🏦 VIRTUAL TRADING SYSTEM INITIALIZED")
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"🤖 Model: Advanced XGBoost (62%+ target)")
        print("="*60)
    
    def calculate_indicators(self, df):
        """Same indicator calculation as training"""
        df = df.copy()
        
        # Basic MAs
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        
        # EMA
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        df['ATR_Pct'] = df['ATR'] / df['Close']
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price Changes
        df['Price_Change_1'] = df['Close'].pct_change(1)
        df['Price_Change_5'] = df['Close'].pct_change(5)
        
        return df
    
    def get_signal(self, symbol="BTCUSDT"):
        """Get AI signal from trained model"""
        try:
            # Get recent klines
            klines = self.client.get_klines(symbol=symbol, interval='15m', limit=200)
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades',
                'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
            ])
            
            # Convert to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Get latest row
            latest = df.iloc[-1]
            
            # Prepare features (simplified version - in production use all features)
            feature_values = []
            for feature in self.features[:20]:  # Use top 20 features for speed
                if feature in latest.index:
                    val = latest[feature]
                    if pd.isna(val):
                        val = 0
                    feature_values.append(val)
                else:
                    feature_values.append(0)
            
            # Predict probability
            prob = self.model.predict_proba([feature_values])[0][1]
            
            # Determine signal with confidence thresholds
            if prob > 0.65:
                return "BUY", prob
            elif prob < 0.35:
                return "SELL", prob
            else:
                return "HOLD", prob
                
        except Exception as e:
            print(f"Error getting signal: {e}")
            return "HOLD", 0.5
    
    def calculate_position_size(self, signal_confidence, current_price):
        """Smart position sizing based on confidence and risk"""
        # Base position: 1% of balance per trade
        base_size = self.balance * 0.01
        
        # Adjust by confidence
        confidence_multiplier = (signal_confidence - 0.5) * 4  # 0.65 -> 0.6x, 0.8 -> 1.2x
        position_value = base_size * (1 + confidence_multiplier)
        
        # Max position size: 5% of balance
        position_value = min(position_value, self.balance * 0.05)
        
        # Calculate quantity
        quantity = position_value / current_price
        
        return quantity, position_value
    
    def execute_trade(self, signal, confidence, price):
        """Execute virtual trade"""
        if signal == "BUY" and not any(p['side'] == 'BUY' for p in self.positions):
            quantity, position_value = self.calculate_position_size(confidence, price)
            
            if position_value <= self.balance:
                trade = {
                    'id': len(self.trades) + 1,
                    'side': 'BUY',
                    'price': price,
                    'quantity': quantity,
                    'value': position_value,
                    'timestamp': datetime.now(),
                    'confidence': confidence,
                    'stop_loss': price * 0.98,  # -2%
                    'take_profit': price * 1.04  # +4%
                }
                
                self.positions.append(trade)
                self.balance -= position_value
                
                print(f"\n🔵 BUY EXECUTED")
                print(f"   Price: ${price:.2f}")
                print(f"   Quantity: {quantity:.4f}")
                print(f"   Value: ${position_value:,.2f}")
                print(f"   Remaining Balance: ${self.balance:,.2f}")
                print(f"   Confidence: {confidence:.1%}")
                
                return trade
                
        elif signal == "SELL" and any(p['side'] == 'BUY' for p in self.positions):
            # Find open BUY position
            buy_position = next(p for p in self.positions if p['side'] == 'BUY')
            
            # Calculate P&L
            pnl = (price - buy_position['price']) * buy_position['quantity']
            pnl_pct = (price / buy_position['price'] - 1) * 100
            
            # Update balance
            self.balance += (buy_position['quantity'] * price)
            
            # Record trade
            close_trade = {
                'id': buy_position['id'],
                'entry_price': buy_position['price'],
                'exit_price': price,
                'quantity': buy_position['quantity'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'entry_time': buy_position['timestamp'],
                'exit_time': datetime.now(),
                'confidence': buy_position['confidence']
            }
            
            self.trades.append(close_trade)
            self.positions.remove(buy_position)
            
            # Update performance
            self.performance['total_trades'] += 1
            if pnl > 0:
                self.performance['winning_trades'] += 1
                print(f"\n🟢 PROFIT TRADE")
            else:
                self.performance['losing_trades'] += 1
                print(f"\n🔴 LOSS TRADE")
            
            self.performance['total_pnl'] += pnl
            
            # Update peak balance for drawdown
            if self.balance > self.performance['peak_balance']:
                self.performance['peak_balance'] = self.balance
            else:
                drawdown = (self.performance['peak_balance'] - self.balance) / self.performance['peak_balance']
                if drawdown > self.performance['max_drawdown']:
                    self.performance['max_drawdown'] = drawdown
            
            print(f"   Entry: ${buy_position['price']:.2f}")
            print(f"   Exit: ${price:.2f}")
            print(f"   P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
            print(f"   New Balance: ${self.balance:,.2f}")
            
            return close_trade
        
        return None
    
    def check_exits(self, current_price):
        """Check stop loss and take profit for open positions"""
        exits = []
        
        for position in self.positions[:]:
            if position['side'] == 'BUY':
                # Check stop loss
                if current_price <= position['stop_loss']:
                    # Execute stop loss
                    pnl = (current_price - position['price']) * position['quantity']
                    self.balance += (position['quantity'] * current_price)
                    
                    close_trade = {
                        'id': position['id'],
                        'entry_price': position['price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'pnl_pct': (current_price / position['price'] - 1) * 100,
                        'entry_time': position['timestamp'],
                        'exit_time': datetime.now(),
                        'exit_reason': 'STOP_LOSS',
                        'confidence': position['confidence']
                    }
                    
                    self.trades.append(close_trade)
                    self.positions.remove(position)
                    exits.append(close_trade)
                    
                    print(f"\n⚠️ STOP LOSS HIT")
                    print(f"   Loss: ${pnl:+,.2f}")
                    
                # Check take profit
                elif current_price >= position['take_profit']:
                    pnl = (current_price - position['price']) * position['quantity']
                    self.balance += (position['quantity'] * current_price)
                    
                    close_trade = {
                        'id': position['id'],
                        'entry_price': position['price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'pnl_pct': (current_price / position['price'] - 1) * 100,
                        'entry_time': position['timestamp'],
                        'exit_time': datetime.now(),
                        'exit_reason': 'TAKE_PROFIT',
                        'confidence': position['confidence']
                    }
                    
                    self.trades.append(close_trade)
                    self.positions.remove(position)
                    exits.append(close_trade)
                    
                    print(f"\n🎯 TAKE PROFIT HIT")
                    print(f"   Profit: ${pnl:+,.2f}")
        
        return exits
    
    def get_performance_report(self):
        """Generate detailed performance report"""
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = (self.performance['winning_trades'] / self.performance['total_trades'] * 100) if self.performance['total_trades'] > 0 else 0
        
        # Calculate Sharpe Ratio (simplified)
        if len(self.trades) > 0:
            returns = [t['pnl_pct'] for t in self.trades]
            sharpe = np.mean(returns) / (np.std(returns) + 0.0001) * np.sqrt(252)
        else:
            sharpe = 0
        
        report = f"""
        {'='*60}
        📊 VIRTUAL TRADING PERFORMANCE REPORT
        {'='*60}
        
        💰 ACCOUNT SUMMARY
        Initial Balance:    ${self.initial_balance:,.2f}
        Current Balance:    ${self.balance:,.2f}
        Total P&L:          ${self.performance['total_pnl']:+,.2f}
        Total Return:       {total_return:+.2f}%
        
        📈 TRADE STATISTICS
        Total Trades:       {self.performance['total_trades']}
        Winning Trades:     {self.performance['winning_trades']}
        Losing Trades:      {self.performance['losing_trades']}
        Win Rate:           {win_rate:.1f}%
        
        🎯 RISK METRICS
        Max Drawdown:       {self.performance['max_drawdown']:.2%}
        Sharpe Ratio:       {sharpe:.2f}
        
        🏦 POSITIONS
        Open Positions:     {len(self.positions)}
        Open Value:         ${sum(p['value'] for p in self.positions):,.2f}
        
        {'='*60}
        """
        
        return report
    
    def save_trades(self, filename="virtual_trades.json"):
        """Save trade history to file"""
        with open(filename, 'w') as f:
            json.dump({
                'initial_balance': self.initial_balance,
                'current_balance': self.balance,
                'trades': [(t['entry_time'].isoformat() if hasattr(t['entry_time'], 'isoformat') else str(t['entry_time']),
                           t['exit_time'].isoformat() if hasattr(t['exit_time'], 'isoformat') else str(t['exit_time']),
                           t['pnl']) for t in self.trades],
                'performance': self.performance
            }, f, indent=2)
        print(f"💾 Trades saved to {filename}")
    
    def run(self, symbol="BTCUSDT", interval=60):
        """Run virtual trading bot"""
        print("\n🚀 Starting Virtual Trading Bot...")
        print(f"📊 Symbol: {symbol}")
        print(f"⏱️  Checking every {interval} seconds")
        print("\nPress Ctrl+C to stop\n")
        
        try:
            while True:
                # Get AI signal
                signal, confidence = self.get_signal(symbol)
                
                # Get current price
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                
                # Check exits for existing positions
                exits = self.check_exits(current_price)
                
                # Execute new trade if no open positions
                if len(self.positions) == 0 and signal != "HOLD":
                    self.execute_trade(signal, confidence, current_price)
                
                # Print status every 10 iterations
                if len(self.trades) % 10 == 0 and len(self.trades) > 0:
                    print(self.get_performance_report())
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
            print(self.get_performance_report())
            self.save_trades()
            print("✅ Virtual trading session ended")

# ==========================================
# RUN VIRTUAL TRADING SYSTEM
# ==========================================
if __name__ == "__main__":
    bot = VirtualTradingBot(initial_balance=500000)
    bot.run(interval=60)  # Check every 60 seconds