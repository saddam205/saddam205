"""
performance_tracker.py
Part of the app/monitoring module.
Tracks system performance, trade metrics, and operational KPIs.
"""

import time
import psutil
import GPUtil
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    gpu_utilization: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeMetrics:
    """Trade performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_holding_time: float  # in hours
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceTracker:
    """
    Comprehensive performance tracking for system and trading metrics
    """
    
    def __init__(self, history_size: int = 10000, 
                 save_interval: int = 3600):
        """
        Initialize performance tracker
        
        Args:
            history_size: Maximum number of metrics to keep
            save_interval: Save interval in seconds
        """
        self.history_size = history_size
        self.save_interval = save_interval
        
        # Metric storage
        self.system_metrics: deque = deque(maxlen=history_size)
        self.trade_metrics: deque = deque(maxlen=history_size)
        self.latency_metrics: deque = deque(maxlen=history_size)
        
        # Current session stats
        self.session_start = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0
        
        # Trade tracking
        self.trades = []
        self.daily_pnl = {}
        
    def record_system_metrics(self):
        """Record current system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk
            disk = psutil.disk_usage('/')
            
            # Network
            net_io = psutil.net_io_counters()
            
            # GPU (if available)
            gpus = GPUtil.getGPUs()
            gpu_util = gpus[0].load * 100 if gpus else None
            gpu_memory = gpus[0].memoryUsed if gpus else None
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                disk_usage_percent=disk.percent,
                network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv,
                gpu_utilization=gpu_util,
                gpu_memory_used=gpu_memory
            )
            
            self.system_metrics.append(metrics)
            
            # Check for anomalies
            if cpu_percent > 80:
                logger.warning(f"High CPU usage: {cpu_percent}%")
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent}%")
                
        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")
    
    def record_trade_metrics(self, trades: List[Dict]) -> TradeMetrics:
        """
        Calculate and record trade metrics
        
        Args:
            trades: List of trade dictionaries
        
        Returns:
            TradeMetrics object
        """
        if not trades:
            return None
        
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0
        
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio
        returns = [t.get('return_pct', 0) for t in trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        cumulative = np.cumsum([t.get('pnl', 0) for t in trades])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max * 100
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Average holding time
        holding_times = []
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                holding_time = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
                holding_times.append(holding_time)
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        metrics = TradeMetrics(
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_holding_time=avg_holding_time
        )
        
        self.trade_metrics.append(metrics)
        self.trades.extend(trades)
        
        # Keep only last history_size trades
        if len(self.trades) > self.history_size:
            self.trades = self.trades[-self.history_size:]
        
        return metrics
    
    def record_latency(self, operation: str, latency_ms: float):
        """
        Record operation latency
        
        Args:
            operation: Operation name
            latency_ms: Latency in milliseconds
        """
        self.latency_metrics.append({
            'operation': operation,
            'latency_ms': latency_ms,
            'timestamp': datetime.now()
        })
        
        self.request_count += 1
        self.total_latency += latency_ms
    
    def record_error(self, error_type: str, error_message: str):
        """
        Record error occurrence
        
        Args:
            error_type: Type of error
            error_message: Error message
        """
        self.error_count += 1
        logger.error(f"Error recorded: {error_type} - {error_message}")
    
    def get_system_health(self) -> Dict:
        """Get current system health status"""
        if not self.system_metrics:
            return {'status': 'unknown'}
        
        latest = self.system_metrics[-1]
        
        health = {
            'status': 'healthy',
            'cpu': latest.cpu_percent,
            'memory': latest.memory_percent,
            'disk': latest.disk_usage_percent,
            'gpu': latest.gpu_utilization,
            'timestamp': latest.timestamp.isoformat()
        }
        
        # Determine health status
        issues = []
        if latest.cpu_percent > 80:
            issues.append('high_cpu')
        if latest.memory_percent > 90:
            issues.append('high_memory')
        if latest.disk_usage_percent > 95:
            issues.append('low_disk_space')
        
        if issues:
            health['status'] = 'warning'
            health['issues'] = issues
        
        return health
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary"""
        # System performance
        if self.system_metrics:
            avg_cpu = np.mean([m.cpu_percent for m in self.system_metrics[-100:]])
            avg_memory = np.mean([m.memory_percent for m in self.system_metrics[-100:]])
        else:
            avg_cpu = avg_memory = 0
        
        # Trading performance
        if self.trade_metrics:
            latest_trade = self.trade_metrics[-1]
            trade_summary = {
                'win_rate': latest_trade.win_rate,
                'total_pnl': latest_trade.total_pnl,
                'sharpe_ratio': latest_trade.sharpe_ratio,
                'max_drawdown': latest_trade.max_drawdown,
                'profit_factor': latest_trade.profit_factor
            }
        else:
            trade_summary = {}
        
        # Latency performance
        if self.latency_metrics:
            avg_latency = np.mean([m['latency_ms'] for m in self.latency_metrics[-1000:]])
            p95_latency = np.percentile([m['latency_ms'] for m in self.latency_metrics[-1000:]], 95)
        else:
            avg_latency = p95_latency = 0
        
        # Session stats
        session_duration = (datetime.now() - self.session_start).total_seconds() / 3600
        
        return {
            'system': {
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'uptime_hours': session_duration
            },
            'trading': trade_summary,
            'performance': {
                'avg_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'requests': self.request_count,
                'errors': self.error_count,
                'error_rate': self.error_count / self.request_count if self.request_count > 0 else 0
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_daily_pnl_report(self) -> Dict:
        """Generate daily P&L report"""
        daily_report = {}
        
        for trade in self.trades:
            trade_date = trade.get('exit_time', datetime.now()).date()
            pnl = trade.get('pnl', 0)
            
            if trade_date not in daily_report:
                daily_report[trade_date] = {
                    'pnl': 0,
                    'trades': 0,
                    'winners': 0
                }
            
            daily_report[trade_date]['pnl'] += pnl
            daily_report[trade_date]['trades'] += 1
            if pnl > 0:
                daily_report[trade_date]['winners'] += 1
        
        # Calculate win rate per day
        for date, stats in daily_report.items():
            stats['win_rate'] = stats['winners'] / stats['trades'] if stats['trades'] > 0 else 0
        
        return daily_report
    
    def reset(self):
        """Reset all metrics"""
        self.system_metrics.clear()
        self.trade_metrics.clear()
        self.latency_metrics.clear()
        self.session_start = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0
        self.trades.clear()
        self.daily_pnl.clear()
        
        logger.info("Performance tracker reset")