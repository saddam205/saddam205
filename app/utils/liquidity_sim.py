"""
liquidity_sim.py
Part of the app/utils module.
Liquidity simulation for realistic market impact modeling.
"""

import numpy as np
import random
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LiquidityProfile(Enum):
    """Liquidity profiles for different market conditions"""
    DEEP = "deep"              # High liquidity, low impact
    NORMAL = "normal"          # Standard liquidity
    THIN = "thin"              # Low liquidity, higher impact
    ILLIQUID = "illiquid"      # Very low liquidity
    FLASH = "flash"            # Rapidly changing liquidity


@dataclass
class LiquidityConfig:
    """Liquidity configuration"""
    base_depth: float  # Order book depth in base currency
    spread_bps: float  # Bid-ask spread in basis points
    impact_factor: float  # Market impact factor
    volatility_factor: float  # Liquidity volatility
    recovery_time_ms: float  # Time to recover after large order


class LiquiditySimulator:
    """
    Simulates market liquidity and order book dynamics.
    Models market impact and slippage for large orders.
    """
    
    def __init__(self, profile: LiquidityProfile = LiquidityProfile.NORMAL):
        """
        Initialize liquidity simulator
        
        Args:
            profile: Liquidity profile to use
        """
        self.profile = profile
        self.config = self._get_config_for_profile(profile)
        self.current_depth = self.config.base_depth
        self.order_history: list = []
        
    def _get_config_for_profile(self, profile: LiquidityProfile) -> LiquidityConfig:
        """Get liquidity configuration for a profile"""
        profiles = {
            LiquidityProfile.DEEP: LiquidityConfig(
                base_depth=10_000_000, spread_bps=1, impact_factor=0.0001, 
                volatility_factor=0.1, recovery_time_ms=100
            ),
            LiquidityProfile.NORMAL: LiquidityConfig(
                base_depth=1_000_000, spread_bps=3, impact_factor=0.0005,
                volatility_factor=0.3, recovery_time_ms=500
            ),
            LiquidityProfile.THIN: LiquidityConfig(
                base_depth=100_000, spread_bps=10, impact_factor=0.002,
                volatility_factor=0.5, recovery_time_ms=2000
            ),
            LiquidityProfile.ILLIQUID: LiquidityConfig(
                base_depth=10_000, spread_bps=50, impact_factor=0.01,
                volatility_factor=0.8, recovery_time_ms=5000
            ),
            LiquidityProfile.FLASH: LiquidityConfig(
                base_depth=500_000, spread_bps=5, impact_factor=0.001,
                volatility_factor=1.5, recovery_time_ms=200
            ),
        }
        return profiles.get(profile, profiles[LiquidityProfile.NORMAL])
    
    def get_slippage(self, order_size_usd: float, side: str) -> float:
        """
        Calculate expected slippage for an order
        
        Args:
            order_size_usd: Order size in USD
            side: 'BUY' or 'SELL'
        
        Returns:
            Slippage as a multiplier (e.g., 0.001 = 0.1%)
        """
        # Base slippage from spread
        base_slippage = self.config.spread_bps / 10000
        
        # Market impact based on order size relative to depth
        impact_ratio = order_size_usd / max(self.current_depth, 1)
        market_impact = self.config.impact_factor * impact_ratio
        
        # Random component (liquidity noise)
        noise = np.random.normal(0, self.config.volatility_factor * 0.0005)
        
        total_slippage = base_slippage + market_impact + abs(noise)
        
        # Update depth after order
        self._update_depth(order_size_usd)
        
        return min(total_slippage, 0.05)  # Cap at 5%
    
    def get_spread(self) -> float:
        """
        Get current bid-ask spread
        
        Returns:
            Spread as a multiplier
        """
        base_spread = self.config.spread_bps / 10000
        noise = np.random.normal(0, self.config.volatility_factor * 0.0002)
        return max(base_spread + noise, 0.0001)
    
    def get_effective_price(self, price: float, order_size_usd: float, side: str) -> float:
        """
        Calculate effective execution price including slippage
        
        Args:
            price: Reference price
            order_size_usd: Order size in USD
            side: 'BUY' or 'SELL'
        
        Returns:
            Effective execution price
        """
        slippage = self.get_slippage(order_size_usd, side)
        
        if side.upper() == 'BUY':
            effective_price = price * (1 + slippage)
        else:
            effective_price = price * (1 - slippage)
        
        return effective_price
    
    def _update_depth(self, order_size_usd: float):
        """
        Update order book depth based on order consumption
        
        Args:
            order_size_usd: Order size in USD
        """
        # Reduce depth by order size
        self.current_depth -= order_size_usd * self.config.impact_factor * 10
        
        # Ensure minimum depth
        self.current_depth = max(self.current_depth, self.config.base_depth * 0.1)
        
        # Record order
        self.order_history.append({
            'size': order_size_usd,
            'depth_after': self.current_depth,
            'timestamp': __import__('time').time()
        })
        
        # Keep only last 1000 orders
        if len(self.order_history) > 1000:
            self.order_history.pop(0)
    
    def recover_depth(self, elapsed_ms: float):
        """
        Recover order book depth over time
        
        Args:
            elapsed_ms: Milliseconds elapsed since last update
        """
        recovery_rate = elapsed_ms / self.config.recovery_time_ms
        recovery_amount = (self.config.base_depth - self.current_depth) * min(0.1, recovery_rate)
        self.current_depth = min(self.config.base_depth, self.current_depth + recovery_amount)
    
    def get_max_order_size(self, max_slippage_bps: float = 10) -> float:
        """
        Calculate maximum order size for a given slippage tolerance
        
        Args:
            max_slippage_bps: Maximum allowed slippage in basis points
        
        Returns:
            Maximum order size in USD
        """
        max_slippage = max_slippage_bps / 10000
        base_slippage = self.config.spread_bps / 10000
        
        if max_slippage <= base_slippage:
            return 0
        
        available_for_impact = max_slippage - base_slippage
        max_size = available_for_impact / self.config.impact_factor * self.current_depth
        
        return max(0, max_size)
    
    def get_statistics(self) -> Dict:
        """
        Get liquidity statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'profile': self.profile.value,
            'current_depth_usd': self.current_depth,
            'base_depth_usd': self.config.base_depth,
            'depth_percentage': (self.current_depth / self.config.base_depth) * 100,
            'spread_bps': self.config.spread_bps,
            'impact_factor': self.config.impact_factor,
            'total_orders': len(self.order_history),
            'avg_order_size': np.mean([o['size'] for o in self.order_history]) if self.order_history else 0
        }
    
    def change_profile(self, profile: LiquidityProfile):
        """Change liquidity profile"""
        self.profile = profile
        self.config = self._get_config_for_profile(profile)
        self.current_depth = self.config.base_depth
        logger.info(f"Liquidity profile changed to {profile.value}")
    
    def set_custom_config(self, base_depth: float, spread_bps: float,
                          impact_factor: float, volatility_factor: float,
                          recovery_time_ms: float):
        """Set custom liquidity configuration"""
        self.config = LiquidityConfig(
            base_depth=base_depth,
            spread_bps=spread_bps,
            impact_factor=impact_factor,
            volatility_factor=volatility_factor,
            recovery_time_ms=recovery_time_ms
        )
        self.current_depth = base_depth
        self.profile = LiquidityProfile.FLASH
        logger.info(f"Custom liquidity config set: depth={base_depth}, spread={spread_bps}bps")