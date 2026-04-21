"""
smart_routing.py
Part of the app/execution module.
Smart order routing for optimal execution across multiple exchanges.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


class Exchange(Enum):
    """Supported exchanges"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BYBIT = "bybit"
    OKX = "okx"


@dataclass
class ExchangeInfo:
    """Exchange information"""
    exchange: Exchange
    base_fee: float
    volume_tier: float
    latency_ms: float
    liquidity_score: float
    reliability_score: float
    available_pairs: List[str]
    current_spread: float = 0.0
    order_book_depth: float = 0.0


@dataclass
class RoutingDecision:
    """Routing decision for an order"""
    exchange: Exchange
    quantity: float
    expected_cost: float
    expected_slippage: float
    estimated_time_ms: float
    reason: str


class SmartRouter:
    """
    Smart order router that optimizes execution across multiple exchanges
    based on liquidity, fees, latency, and current market conditions.
    """
    
    def __init__(self):
        """Initialize smart router with exchange information"""
        self.exchanges: Dict[Exchange, ExchangeInfo] = {}
        self.routing_history: List[Dict] = []
        self.performance_metrics: Dict = {}
        
        # Initialize exchange info (would fetch from APIs)
        self._init_exchanges()
        
    def _init_exchanges(self):
        """Initialize exchange information"""
        self.exchanges = {
            Exchange.BINANCE: ExchangeInfo(
                exchange=Exchange.BINANCE,
                base_fee=0.001,
                volume_tier=0.00075,
                latency_ms=50,
                liquidity_score=0.95,
                reliability_score=0.99,
                available_pairs=['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            ),
            Exchange.COINBASE: ExchangeInfo(
                exchange=Exchange.COINBASE,
                base_fee=0.004,
                volume_tier=0.002,
                latency_ms=100,
                liquidity_score=0.85,
                reliability_score=0.98,
                available_pairs=['BTC-USD', 'ETH-USD']
            ),
            Exchange.KRAKEN: ExchangeInfo(
                exchange=Exchange.KRAKEN,
                base_fee=0.0026,
                volume_tier=0.002,
                latency_ms=80,
                liquidity_score=0.80,
                reliability_score=0.97,
                available_pairs=['XBT/USD', 'ETH/USD']
            ),
            Exchange.BYBIT: ExchangeInfo(
                exchange=Exchange.BYBIT,
                base_fee=0.0005,
                volume_tier=0.0004,
                latency_ms=60,
                liquidity_score=0.75,
                reliability_score=0.95,
                available_pairs=['BTCUSDT', 'ETHUSDT']
            )
        }
        
        # Update current market conditions
        self._update_market_conditions()
    
    def _update_market_conditions(self):
        """Update current market conditions for each exchange"""
        # In production, this would fetch real-time data
        for exchange in self.exchanges.values():
            exchange.current_spread = np.random.uniform(0.0005, 0.003)
            exchange.order_book_depth = np.random.uniform(500000, 5000000)
    
    async def route_order(self, symbol: str, side: str, quantity: float,
                          max_slippage: float = 0.01,
                          prefer_low_latency: bool = False) -> List[RoutingDecision]:
        """
        Route an order across exchanges
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            max_slippage: Maximum acceptable slippage
            prefer_low_latency: Prioritize low latency over cost
        
        Returns:
            List of routing decisions
        """
        # Find exchanges that support this symbol
        eligible_exchanges = self._find_eligible_exchanges(symbol)
        
        if not eligible_exchanges:
            logger.warning(f"No exchanges support {symbol}")
            return []
        
        # Calculate routing scores
        decisions = []
        
        for exchange in eligible_exchanges:
            score = self._calculate_routing_score(
                exchange, quantity, side, max_slippage, prefer_low_latency
            )
            
            if score > 0:
                decisions.append(score)
        
        # Sort by score (higher is better)
        decisions.sort(key=lambda x: x.expected_cost, reverse=False)
        
        # Determine split if necessary
        if len(decisions) > 1 and quantity > 100:
            decisions = self._optimize_split(decisions, quantity)
        
        # Log routing decision
        self.routing_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'total_quantity': quantity,
            'decisions': [(d.exchange.value, d.quantity) for d in decisions]
        })
        
        return decisions
    
    def _find_eligible_exchanges(self, symbol: str) -> List[ExchangeInfo]:
        """Find exchanges that support the given symbol"""
        eligible = []
        
        for exchange in self.exchanges.values():
            # Check if symbol is supported (simplified)
            normalized_symbol = symbol.replace('USDT', 'USDT').replace('-', '')
            if any(normalized_symbol in pair for pair in exchange.available_pairs):
                eligible.append(exchange)
        
        return eligible
    
    def _calculate_routing_score(self, exchange: ExchangeInfo, quantity: float,
                                 side: str, max_slippage: float,
                                 prefer_low_latency: bool) -> RoutingDecision:
        """Calculate routing score for an exchange"""
        # Calculate expected costs
        fee_cost = self._calculate_fee_cost(exchange, quantity)
        slippage_cost = self._calculate_slippage_cost(exchange, quantity, side)
        latency_cost = self._calculate_latency_cost(exchange, prefer_low_latency)
        
        total_cost = fee_cost + slippage_cost + latency_cost
        
        # Check if within max slippage
        if slippage_cost / quantity > max_slippage:
            return None
        
        return RoutingDecision(
            exchange=exchange.exchange,
            quantity=quantity,
            expected_cost=total_cost,
            expected_slippage=slippage_cost,
            estimated_time_ms=exchange.latency_ms,
            reason=self._get_routing_reason(exchange, total_cost)
        )
    
    def _calculate_fee_cost(self, exchange: ExchangeInfo, quantity: float) -> float:
        """Calculate fee cost"""
        # Assume average price of $100 for calculation
        avg_price = 100.0
        fee_rate = min(exchange.base_fee, exchange.volume_tier)  # Use volume tier if applicable
        return quantity * avg_price * fee_rate
    
    def _calculate_slippage_cost(self, exchange: ExchangeInfo, quantity: float,
                                 side: str) -> float:
        """Calculate expected slippage cost"""
        # Slippage depends on order size relative to liquidity
        liquidity_impact = quantity / exchange.order_book_depth if exchange.order_book_depth > 0 else 0.01
        base_slippage = exchange.current_spread / 2
        slippage = base_slippage * (1 + liquidity_impact)
        
        avg_price = 100.0
        return quantity * avg_price * slippage
    
    def _calculate_latency_cost(self, exchange: ExchangeInfo, prefer_low_latency: bool) -> float:
        """Calculate latency cost"""
        if prefer_low_latency:
            # Lower latency is better
            return exchange.latency_ms / 1000 * 0.01
        return 0.0
    
    def _get_routing_reason(self, exchange: ExchangeInfo, total_cost: float) -> str:
        """Get routing reason text"""
        if total_cost < 0.001:
            return f"Lowest cost on {exchange.exchange.value}"
        elif exchange.liquidity_score > 0.9:
            return f"Best liquidity on {exchange.exchange.value}"
        elif exchange.reliability_score > 0.99:
            return f"Highest reliability on {exchange.exchange.value}"
        else:
            return f"Balanced execution on {exchange.exchange.value}"
    
    def _optimize_split(self, decisions: List[RoutingDecision],
                        total_quantity: float) -> List[RoutingDecision]:
        """Optimize order splitting across exchanges"""
        if len(decisions) <= 1:
            return decisions
        
        # Simple optimization: allocate based on liquidity
        total_liquidity = sum(d.expected_slippage for d in decisions)
        optimized = []
        
        for decision in decisions:
            if total_liquidity > 0:
                allocation = total_quantity * (1 - decision.expected_slippage / total_liquidity)
                allocation = max(0.1 * total_quantity, min(allocation, total_quantity * 0.5))
            else:
                allocation = total_quantity / len(decisions)
            
            optimized.append(RoutingDecision(
                exchange=decision.exchange,
                quantity=allocation,
                expected_cost=decision.expected_cost * (allocation / decision.quantity),
                expected_slippage=decision.expected_slippage,
                estimated_time_ms=decision.estimated_time_ms,
                reason=decision.reason
            ))
        
        return optimized
    
    async def execute_routed_orders(self, decisions: List[RoutingDecision],
                                    order_executor) -> List[Dict]:
        """Execute routed orders"""
        results = []
        
        for decision in decisions:
            if decision.quantity <= 0:
                continue
            
            result = await order_executor.execute(
                symbol=None,  # Would need symbol mapping
                side=None,
                quantity=decision.quantity,
                exchange=decision.exchange.value
            )
            
            results.append({
                'exchange': decision.exchange.value,
                'quantity': decision.quantity,
                'result': result
            })
        
        return results
    
    def get_exchange_stats(self) -> Dict[str, Dict]:
        """Get exchange performance statistics"""
        stats = {}
        
        for exchange in self.exchanges.values():
            # Calculate stats from routing history
            exchange_routes = [
                r for r in self.routing_history
                if any(d.exchange == exchange.exchange for d in r.get('decisions', []))
            ]
            
            stats[exchange.exchange.value] = {
                'total_routes': len(exchange_routes),
                'avg_expected_cost': np.mean([r.get('expected_cost', 0) for r in exchange_routes]) if exchange_routes else 0,
                'liquidity_score': exchange.liquidity_score,
                'reliability_score': exchange.reliability_score,
                'avg_latency': exchange.latency_ms
            }
        
        return stats
    
    def update_exchange_metrics(self, exchange: Exchange, metrics: Dict):
        """Update exchange performance metrics"""
        if exchange.value in self.exchanges:
            for key, value in metrics.items():
                if hasattr(self.exchanges[exchange.value], key):
                    setattr(self.exchanges[exchange.value], key, value)
    
    def get_optimal_exchange(self, symbol: str, side: str) -> Optional[Exchange]:
        """Get the optimal exchange for immediate execution"""
        decisions = asyncio.run(self.route_order(symbol, side, 1.0, max_slippage=0.01))
        
        if decisions:
            return decisions[0].exchange
        
        return None