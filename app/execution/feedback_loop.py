"""
feedback_loop.py
Part of the app/execution module.
Execution feedback loop for learning and optimization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Execution performance metrics"""
    order_id: str
    symbol: str
    expected_price: float
    executed_price: float
    slippage: float
    latency_ms: float
    market_impact: float
    timestamp: datetime
    success: bool
    exchange: str = None
    order_size: float = 0.0
    
    @property
    def slippage_bps(self) -> float:
        """Slippage in basis points"""
        return self.slippage * 10000


@dataclass
class FeedbackModel:
    """Model for execution feedback"""
    exchange: str
    symbol_pattern: str
    avg_slippage: float = 0.0
    avg_latency: float = 0.0
    success_rate: float = 1.0
    market_impact_coefficient: float = 0.001
    last_updated: datetime = field(default_factory=datetime.now)
    samples: int = 0
    
    def update(self, metrics: ExecutionMetrics):
        """Update model with new execution data"""
        alpha = 0.1  # Learning rate
        
        self.avg_slippage = (1 - alpha) * self.avg_slippage + alpha * metrics.slippage
        self.avg_latency = (1 - alpha) * self.avg_latency + alpha * metrics.latency_ms
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1 if metrics.success else 0)
        
        self.samples += 1
        self.last_updated = datetime.now()
    
    def predict_slippage(self, order_size: float) -> float:
        """Predict slippage for an order"""
        base_slippage = self.avg_slippage
        size_impact = order_size * self.market_impact_coefficient
        return base_slippage + size_impact


class ExecutionFeedbackLoop:
    """
    Execution feedback loop that learns from past executions
    to optimize future order routing and execution.
    """
    
    def __init__(self, lookback_days: int = 30, update_interval_seconds: int = 60):
        """
        Initialize execution feedback loop
        
        Args:
            lookback_days: Number of days to keep historical data
            update_interval_seconds: How often to update models
        """
        self.lookback_days = lookback_days
        self.update_interval = update_interval_seconds
        self.execution_history: deque = deque(maxlen=10000)
        self.feedback_models: Dict[str, FeedbackModel] = {}
        self.is_running = False
        
        # Performance tracking
        self.performance_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'total_slippage': 0.0,
            'avg_slippage_bps': 0.0,
            'avg_latency_ms': 0.0,
            'improvement_rate': 0.0
        }
        
        # Initialize models
        self._init_models()
        
    def _init_models(self):
        """Initialize feedback models for different exchanges"""
        exchanges = ['binance', 'coinbase', 'kraken', 'bybit', 'okx']
        symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']
        
        for exchange in exchanges:
            for symbol in symbols:
                model_key = f"{exchange}_{symbol}"
                self.feedback_models[model_key] = FeedbackModel(
                    exchange=exchange,
                    symbol_pattern=symbol
                )
    
    def record_execution(self, metrics: ExecutionMetrics):
        """
        Record an execution for feedback learning
        
        Args:
            metrics: Execution metrics
        """
        self.execution_history.append(metrics)
        self.performance_metrics['total_executions'] += 1
        
        if metrics.success:
            self.performance_metrics['successful_executions'] += 1
        
        self.performance_metrics['total_slippage'] += metrics.slippage
        self.performance_metrics['avg_slippage_bps'] = (
            self.performance_metrics['total_slippage'] / 
            self.performance_metrics['total_executions'] * 10000
        )
        self.performance_metrics['avg_latency_ms'] = (
            self.performance_metrics['avg_latency_ms'] * 0.95 + 
            metrics.latency_ms * 0.05
        )
        
        # Update specific model
        model_key = self._get_model_key(metrics.exchange, metrics.symbol)
        if model_key in self.feedback_models:
            self.feedback_models[model_key].update(metrics)
        
        logger.debug(f"Recorded execution: {metrics.symbol} on {metrics.exchange} - slippage: {metrics.slippage_bps:.1f}bps")
    
    def _get_model_key(self, exchange: str, symbol: str) -> str:
        """Get model key from exchange and symbol"""
        # Extract base symbol (remove USDT, USD, etc.)
        base_symbol = symbol.replace('USDT', '').replace('USD', '').replace('USDC', '')
        return f"{exchange}_{base_symbol}"
    
    def get_execution_advice(self, exchange: str, symbol: str, 
                             order_size: float) -> Dict:
        """
        Get execution advice based on historical feedback
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            order_size: Order size
        
        Returns:
            Execution advice dictionary
        """
        model_key = self._get_model_key(exchange, symbol)
        
        if model_key not in self.feedback_models:
            return {
                'expected_slippage': 0.001,  # 10 bps default
                'confidence': 0.5,
                'suggested_slices': 1,
                'advice': 'Insufficient data - use conservative execution'
            }
        
        model = self.feedback_models[model_key]
        
        # Predict slippage
        predicted_slippage = model.predict_slippage(order_size)
        
        # Calculate confidence based on sample size
        confidence = min(0.95, model.samples / 100)
        
        # Suggest slice count based on order size
        suggested_slices = 1
        if order_size > 10 and predicted_slippage > 0.002:
            suggested_slices = max(2, min(10, int(order_size / 5)))
        
        # Generate advice
        if predicted_slippage < 0.0005:
            advice = "Favorable conditions - execute immediately"
        elif predicted_slippage < 0.001:
            advice = "Normal conditions - standard execution recommended"
        elif predicted_slippage < 0.002:
            advice = "Use TWAP to reduce impact"
        else:
            advice = "High slippage expected - consider limit orders"
        
        return {
            'expected_slippage': predicted_slippage,
            'expected_slippage_bps': predicted_slippage * 10000,
            'confidence': confidence,
            'suggested_slices': suggested_slices,
            'advice': advice,
            'model_samples': model.samples,
            'historical_avg_slippage': model.avg_slippage * 10000
        }
    
    def get_best_exchange(self, symbol: str, order_size: float) -> Optional[str]:
        """
        Find the best exchange for a given symbol and order size
        
        Args:
            symbol: Trading symbol
            order_size: Order size
        
        Returns:
            Best exchange name or None
        """
        best_exchange = None
        best_score = float('inf')
        
        for model in self.feedback_models.values():
            if symbol in model.symbol_pattern:
                predicted_slippage = model.predict_slippage(order_size)
                score = predicted_slippage * (1 - model.success_rate)
                
                if score < best_score:
                    best_score = score
                    best_exchange = model.exchange
        
        return best_exchange
    
    def calculate_improvement_rate(self) -> float:
        """
        Calculate improvement rate over time
        
        Returns:
            Improvement rate (0-1)
        """
        if len(self.execution_history) < 100:
            return 0.0
        
        # Split history into two halves
        history_list = list(self.execution_history)
        mid_point = len(history_list) // 2
        
        first_half = history_list[:mid_point]
        second_half = history_list[mid_point:]
        
        # Calculate average slippage for each half
        first_avg = np.mean([m.slippage for m in first_half]) if first_half else 0
        second_avg = np.mean([m.slippage for m in second_half]) if second_half else 0
        
        # Calculate improvement
        if first_avg > 0:
            improvement = (first_avg - second_avg) / first_avg
        else:
            improvement = 0
        
        self.performance_metrics['improvement_rate'] = max(0, min(1, improvement))
        
        return self.performance_metrics['improvement_rate']
    
    def get_performance_report(self) -> str:
        """Generate performance report"""
        self.calculate_improvement_rate()
        
        report = []
        report.append("=" * 60)
        report.append("EXECUTION FEEDBACK LOOP REPORT")
        report.append("=" * 60)
        report.append(f"Total Executions: {self.performance_metrics['total_executions']}")
        report.append(f"Success Rate: {self.performance_metrics['successful_executions'] / max(1, self.performance_metrics['total_executions']):.1%}")
        report.append(f"Avg Slippage: {self.performance_metrics['avg_slippage_bps']:.1f} bps")
        report.append(f"Avg Latency: {self.performance_metrics['avg_latency_ms']:.1f} ms")
        report.append(f"Improvement Rate: {self.performance_metrics['improvement_rate']:.1%}")
        report.append("")
        report.append("Exchange Performance:")
        
        # Group by exchange
        exchange_stats = {}
        for metrics in self.execution_history:
            if metrics.exchange not in exchange_stats:
                exchange_stats[metrics.exchange] = {'slippage': [], 'latency': [], 'success': []}
            exchange_stats[metrics.exchange]['slippage'].append(metrics.slippage)
            exchange_stats[metrics.exchange]['latency'].append(metrics.latency_ms)
            exchange_stats[metrics.exchange]['success'].append(1 if metrics.success else 0)
        
        for exchange, stats in exchange_stats.items():
            avg_slippage = np.mean(stats['slippage']) * 10000 if stats['slippage'] else 0
            avg_latency = np.mean(stats['latency']) if stats['latency'] else 0
            success_rate = np.mean(stats['success']) if stats['success'] else 0
            
            report.append(f"  {exchange.upper()}: {avg_slippage:.1f}bps slippage, {avg_latency:.0f}ms, {success_rate:.1%} success")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def reset(self):
        """Reset feedback loop data"""
        self.execution_history.clear()
        self.performance_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'total_slippage': 0.0,
            'avg_slippage_bps': 0.0,
            'avg_latency_ms': 0.0,
            'improvement_rate': 0.0
        }
        self._init_models()
        
        logger.info("Execution feedback loop reset")
    
    async def start_monitoring(self):
        """Start monitoring and learning loop"""
        self.is_running = True
        
        while self.is_running:
            try:
                # Update performance metrics
                self.calculate_improvement_rate()
                
                # Log periodic report
                if len(self.execution_history) % 100 == 0 and len(self.execution_history) > 0:
                    logger.info(f"Execution feedback report:\n{self.get_performance_report()}")
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Feedback loop error: {e}")
                await asyncio.sleep(5)
    
    def stop_monitoring(self):
        """Stop monitoring loop"""
        self.is_running = False
    
    def get_recommended_execution_params(self, symbol: str, order_size: float,
                                        exchange: str = None) -> Dict:
        """
        Get recommended execution parameters
        
        Args:
            symbol: Trading symbol
            order_size: Order size
            exchange: Specific exchange (optional)
        
        Returns:
            Recommended execution parameters
        """
        if exchange:
            advice = self.get_execution_advice(exchange, symbol, order_size)
            return {
                'exchange': exchange,
                'execution_algorithm': 'TWAP' if advice['suggested_slices'] > 1 else 'MARKET',
                'slices': advice['suggested_slices'],
                'expected_slippage': advice['expected_slippage_bps'],
                'confidence': advice['confidence']
            }
        else:
            # Find best exchange
            best_exchange = self.get_best_exchange(symbol, order_size)
            if best_exchange:
                return self.get_recommended_execution_params(symbol, order_size, best_exchange)
            else:
                return {
                    'exchange': 'default',
                    'execution_algorithm': 'MARKET',
                    'slices': 1,
                    'expected_slippage': 10,
                    'confidence': 0.3
                }