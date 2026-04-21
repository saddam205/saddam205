"""
Risk Management System
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """Professional risk management"""
    
    def __init__(self):
        self.daily_pnl = []
        self.positions = []
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.consecutive_losses = 0
        self.returns_history = []
        
        # Risk limits
        self.limits = {
            "max_position_pct": 0.10,
            "max_portfolio_risk": 0.20,
            "max_daily_loss": 0.05,
            "max_drawdown": 0.15,
            "max_consecutive_losses": 5
        }
    
    def check_position(self, symbol: str, size: float, price: float, capital: float) -> Tuple[bool, str]:
        """Check if position is within risk limits"""
        position_value = size * price
        position_pct = position_value / capital
        
        if position_pct > self.limits["max_position_pct"]:
            return False, f"Position size {position_pct:.1%} exceeds limit"
        
        if self._check_daily_loss():
            return False, "Daily loss limit reached"
        
        if self.current_drawdown > self.limits["max_drawdown"]:
            return False, f"Drawdown {self.current_drawdown:.1%} exceeds limit"
        
        if self.consecutive_losses >= self.limits["max_consecutive_losses"]:
            return False, "Consecutive losses limit reached"
        
        return True, "OK"
    
    def update_position(self, symbol: str, size: float, price: float, side: str, pnl: float):
        """Update position after trade"""
        if side == "BUY":
            self.positions.append({
                "symbol": symbol,
                "size": size,
                "entry_price": price,
                "entry_time": datetime.now()
            })
        else:
            self.positions = [p for p in self.positions if p["symbol"] != symbol]
        
        if pnl != 0:
            self._update_pnl(pnl)
    
    def _update_pnl(self, pnl: float):
        """Update P&L tracking"""
        today = datetime.now().date()
        
        if not self.daily_pnl or self.daily_pnl[-1]["date"] != today:
            self.daily_pnl.append({"date": today, "pnl": 0})
        
        self.daily_pnl[-1]["pnl"] += pnl
        
        if pnl < 0:
            self.consecutive_losses += 1
            self.returns_history.append(pnl)
        else:
            self.consecutive_losses = 0
            self.returns_history.append(pnl)
        
        # Keep last 100 returns
        if len(self.returns_history) > 100:
            self.returns_history.pop(0)
    
    def _check_daily_loss(self) -> bool:
        """Check if daily loss limit exceeded"""
        if not self.daily_pnl:
            return False
        
        today_pnl = self.daily_pnl[-1]["pnl"]
        daily_loss_pct = abs(today_pnl) / 100000  # Assuming $100k capital
        
        return daily_loss_pct > self.limits["max_daily_loss"]
    
    def calculate_var(self, returns: np.ndarray = None, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if returns is None:
            returns = np.array(self.returns_history) if self.returns_history else np.array([0])
        
        if len(returns) < 30:
            return 0.05
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: np.ndarray = None, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR"""
        if returns is None:
            returns = np.array(self.returns_history) if self.returns_history else np.array([0])
        
        var = self.calculate_var(returns, confidence)
        tail_returns = returns[returns <= var]
        
        return np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    async def filter_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Filter signals based on risk constraints
        
        Args:
            signals: List of trading signals
        
        Returns:
            Filtered signals that pass risk checks
        """
        filtered = []
        
        for signal in signals:
            # Check if we already have a position in this symbol
            has_position = any(p.get('symbol') == signal['symbol'] for p in self.positions)
            
            if has_position:
                logger.debug(f"Skipping {signal['symbol']} - position already exists")
                continue
            
            # Check confidence threshold
            if signal.get('confidence', 0) < 0.6:
                logger.debug(f"Skipping {signal['symbol']} - low confidence: {signal.get('confidence')}")
                continue
            
            # Check daily loss limit
            if self._check_daily_loss():
                logger.warning("Daily loss limit reached - skipping all signals")
                break
            
            filtered.append(signal)
        
        return filtered
    
    async def calculate_position_size(self, symbol: str, confidence: float, 
                                      portfolio_value: float) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            symbol: Asset symbol
            confidence: Signal confidence (0-1)
            portfolio_value: Total portfolio value
        
        Returns:
            Position size in quantity
        """
        # Base position size as percentage of portfolio
        base_pct = 0.05  # 5% base position
        
        # Adjust by confidence
        size_pct = base_pct * confidence
        
        # Apply max limit
        size_pct = min(size_pct, self.limits["max_position_pct"])
        
        # Calculate position value
        position_value = portfolio_value * size_pct
        
        # Get current price (simplified)
        price = 100  # Default price, should be fetched from market
        
        # Calculate quantity
        quantity = position_value / price
        
        return quantity
    
    def get_risk_report(self) -> Dict:
        """Get comprehensive risk report"""
        returns = np.array(self.returns_history) if self.returns_history else np.array([0])
        
        return {
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "consecutive_losses": self.consecutive_losses,
            "daily_pnl": self.daily_pnl[-5:] if self.daily_pnl else [],
            "limits": self.limits,
            "open_positions": len(self.positions),
            "var_95": self.calculate_var(returns, 0.95),
            "var_99": self.calculate_var(returns, 0.99),
            "cvar_95": self.calculate_cvar(returns, 0.95)
        }
    
    def get_current_limits(self) -> Dict:
        """Get current risk limits"""
        return self.limits
    
    def update_limits(self, limits: Dict) -> Dict:
        """Update risk limits"""
        self.limits.update(limits)
        logger.info(f"Risk limits updated: {limits}")
        return self.limits
    
    def update_drawdown(self, current_equity: float, peak_equity: float):
        """Update drawdown metrics"""
        if current_equity > peak_equity:
            peak_equity = current_equity
        
        self.current_drawdown = (peak_equity - current_equity) / peak_equity * 100
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        return self.current_drawdown