"""
position_sizer.py
Dynamic position sizing based on confidence, volatility, and risk.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from decimal import Decimal


class DynamicPositionSizer:
    """Dynamic position sizing with multiple strategies"""
    
    def __init__(self, max_position_pct: float = 0.25, min_position_pct: float = 0.01):
        """
        Initialize position sizer
        
        Args:
            max_position_pct: Maximum position size as percentage of capital
            min_position_pct: Minimum position size as percentage of capital
        """
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        
        # Risk parameters
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.max_risk_per_day = 0.05  # 5% max daily risk
        
        # Volatility adjustment
        self.volatility_scale = 1.0
        self.target_volatility = 0.02  # Target 2% daily volatility
        
        # Kelly parameters
        self.kelly_fraction = 0.25  # Use 25% of Kelly for safety
        self.win_rate_history = []
        self.avg_win_loss_ratio = 1.5
        
    def calculate_position(self, balance: float, confidence: float, 
                          volatility: float = 0.02, current_price: float = None) -> Tuple[float, float, Dict]:
        """
        Calculate position size using multiple methods
        
        Args:
            balance: Account balance or capital
            confidence: Signal confidence (0 to 1)
            volatility: Asset volatility (daily standard deviation)
            current_price: Current asset price (optional)
        
        Returns:
            Tuple of (position_value, quantity, sizing_info)
        """
        # Method 1: Risk-based sizing
        risk_based_size = self._risk_based_sizing(balance, confidence)
        
        # Method 2: Kelly Criterion sizing
        kelly_size = self._kelly_sizing(balance, confidence)
        
        # Method 3: Volatility-adjusted sizing
        vol_adjusted_size = self._volatility_adjusted_sizing(balance, volatility, confidence)
        
        # Combine methods with weighted average
        position_value = (
            risk_based_size * 0.4 +
            kelly_size * 0.3 +
            vol_adjusted_size * 0.3
        )
        
        # Apply bounds
        position_value = self._apply_bounds(position_value, balance)
        
        # Calculate quantity if price provided
        quantity = position_value / current_price if current_price else 0
        
        # Prepare sizing info
        sizing_info = {
            'method': 'combined',
            'risk_based_size': risk_based_size,
            'kelly_size': kelly_size,
            'vol_adjusted_size': vol_adjusted_size,
            'final_size': position_value,
            'percentage': position_value / balance if balance > 0 else 0,
            'confidence': confidence,
            'volatility': volatility
        }
        
        return position_value, quantity, sizing_info
    
    def _risk_based_sizing(self, balance: float, confidence: float) -> float:
        """
        Calculate position size based on fixed risk per trade
        
        Formula: Position = (Risk per trade * Balance) / (Stop Loss % * Confidence)
        """
        risk_amount = balance * self.risk_per_trade
        
        # Assume 2% stop loss for base calculation
        stop_loss_pct = 0.02
        
        # Adjust for confidence
        position_value = (risk_amount / stop_loss_pct) * confidence
        
        return position_value
    
    def _kelly_sizing(self, balance: float, confidence: float) -> float:
        """
        Calculate position size using Kelly Criterion
        
        Kelly % = (p * b - q) / b
        where p = win probability, b = win/loss ratio, q = loss probability
        """
        # Convert confidence to win probability
        win_prob = confidence
        
        # Calculate Kelly fraction
        kelly = (win_prob * self.avg_win_loss_ratio - (1 - win_prob)) / self.avg_win_loss_ratio
        
        # Apply safety factor and bounds
        kelly = max(0, min(kelly, 0.25))  # Limit to 25%
        
        position_value = balance * kelly * self.kelly_fraction
        
        return position_value
    
    def _volatility_adjusted_sizing(self, balance: float, volatility: float, confidence: float) -> float:
        """
        Adjust position size based on volatility
        
        Higher volatility = smaller positions
        """
        # Calculate volatility adjustment factor
        vol_factor = self.target_volatility / max(volatility, 0.005)
        vol_factor = max(0.3, min(vol_factor, 2.0))  # Limit adjustment
        
        # Base size on confidence
        base_size = balance * (confidence * 0.15)  # Max 15% of capital
        
        # Apply volatility adjustment
        position_value = base_size * vol_factor
        
        return position_value
    
    def _apply_bounds(self, position_value: float, balance: float) -> float:
        """Apply min/max position bounds"""
        max_position = balance * self.max_position_pct
        min_position = balance * self.min_position_pct
        
        return max(min_position, min(position_value, max_position))
    
    def calculate_position_by_investment(self, investment_amount: float, 
                                         balance: float, confidence: float) -> Tuple[float, str]:
        """
        Calculate position size based on fixed investment amount
        
        Args:
            investment_amount: User-specified investment amount
            balance: Account balance
            confidence: Signal confidence
        
        Returns:
            Tuple of (position_value, reason)
        """
        # Check if investment amount is valid
        if investment_amount <= 0:
            return 0, "Invalid investment amount"
        
        # Check against max position size
        max_position = balance * self.max_position_pct
        if investment_amount > max_position:
            reason = f"Investment exceeds max position size of ${max_position:.2f}"
            return max_position, reason
        
        # Check against min position size
        min_position = balance * self.min_position_pct
        if investment_amount < min_position:
            reason = f"Investment below minimum position size of ${min_position:.2f}"
            return min_position, reason
        
        # Check risk per trade
        risk_amount = investment_amount * 0.02  # Assume 2% stop loss
        max_risk = balance * self.risk_per_trade
        
        if risk_amount > max_risk:
            adjusted_size = max_risk / 0.02
            reason = f"Risk per trade limit: reduced to ${adjusted_size:.2f}"
            return adjusted_size, reason
        
        return investment_amount, "Investment amount accepted"
    
    def update_kelly_parameters(self, win_rate: float, avg_win_loss_ratio: float):
        """
        Update Kelly parameters based on historical performance
        
        Args:
            win_rate: Historical win rate (0 to 1)
            avg_win_loss_ratio: Average win/loss ratio
        """
        self.win_rate_history.append(win_rate)
        
        # Keep only last 100 trades
        if len(self.win_rate_history) > 100:
            self.win_rate_history.pop(0)
        
        # Update moving average
        self.avg_win_loss_ratio = avg_win_loss_ratio
    
    def get_risk_metrics(self, positions: Dict) -> Dict:
        """
        Calculate risk metrics for current positions
        
        Args:
            positions: Dictionary of current positions
        
        Returns:
            Risk metrics dictionary
        """
        total_exposure = sum(p.get('value', 0) for p in positions.values())
        total_capital = 0  # This should come from portfolio
        
        if total_capital == 0:
            return {'error': 'No capital data'}
        
        # Calculate metrics
        exposure_ratio = total_exposure / total_capital
        diversification_score = min(1.0, len(positions) / 10)  # 10 positions = full diversification
        
        # Calculate concentration risk
        max_position = max([p.get('value', 0) for p in positions.values()], default=0)
        concentration_risk = max_position / total_exposure if total_exposure > 0 else 0
        
        # Calculate correlation risk (simplified)
        correlation_risk = 0.5  # Placeholder for actual correlation calculation
        
        return {
            'total_exposure': total_exposure,
            'exposure_ratio': exposure_ratio,
            'diversification_score': diversification_score,
            'concentration_risk': concentration_risk,
            'correlation_risk': correlation_risk,
            'number_of_positions': len(positions),
            'max_position_size': max_position
        }
    
    def calculate_stop_loss(self, entry_price: float, confidence: float, 
                           volatility: float, stop_loss_type: str = 'atr') -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            confidence: Signal confidence
            volatility: Asset volatility
            stop_loss_type: Type of stop loss ('atr', 'percent', 'volatility')
        
        Returns:
            Stop loss price
        """
        if stop_loss_type == 'atr':
            # Use 2x ATR for stop loss (ATR should be passed or calculated)
            atr = volatility  # Assuming volatility is ATR
            stop_distance = atr * 2
            stop_price = entry_price - stop_distance
        
        elif stop_loss_type == 'percent':
            # Percentage-based stop loss
            stop_pct = 0.02 * (1 + (1 - confidence))  # Lower confidence = wider stop
            stop_distance = entry_price * stop_pct
            stop_price = entry_price - stop_distance
        
        else:  # volatility-based
            # Wider stops in high volatility
            vol_adjustment = 1 + volatility / 0.02
            stop_pct = 0.02 * vol_adjustment
            stop_distance = entry_price * stop_pct
            stop_price = entry_price - stop_distance
        
        # Ensure stop loss is positive
        return max(stop_price, entry_price * 0.95)  # Max 5% loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                              risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit price based on risk-reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_reward_ratio: Risk-reward ratio
        
        Returns:
            Take profit price
        """
        risk = entry_price - stop_loss
        reward = risk * risk_reward_ratio
        
        take_profit = entry_price + reward
        
        return take_profit