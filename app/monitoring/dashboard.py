"""
dashboard.py
Part of the app/monitoring module.
Dashboard management and health check endpoints.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import logging

logger = logging.getLogger(__name__)


class DashboardManager:
    """
    Manages dashboard data aggregation and health checks
    """
    
    def __init__(self):
        """Initialize dashboard manager"""
        self.health_status = {
            'status': 'healthy',
            'checks': {},
            'timestamp': datetime.now()
        }
        self.metrics_cache = {}
        self.last_update = None
        self.update_interval = 30  # seconds
        
    def register_health_check(self, name: str, check_func: callable):
        """
        Register a health check function
        
        Args:
            name: Health check name
            check_func: Async function that returns (status, message)
        """
        self.health_status['checks'][name] = {
            'function': check_func,
            'status': 'unknown',
            'message': '',
            'last_check': None
        }
        logger.info(f"Registered health check: {name}")
    
    async def run_health_checks(self):
        """Run all registered health checks"""
        all_healthy = True
        
        for name, check_data in self.health_status['checks'].items():
            try:
                status, message = await check_data['function']()
                check_data['status'] = status
                check_data['message'] = message
                check_data['last_check'] = datetime.now()
                
                if status != 'healthy':
                    all_healthy = False
                    
            except Exception as e:
                check_data['status'] = 'unhealthy'
                check_data['message'] = str(e)
                check_data['last_check'] = datetime.now()
                all_healthy = False
        
        self.health_status['status'] = 'healthy' if all_healthy else 'degraded'
        self.health_status['timestamp'] = datetime.now()
        
        return self.health_status
    
    def get_health_status(self) -> Dict:
        """Get current health status"""
        return {
            'status': self.health_status['status'],
            'timestamp': self.health_status['timestamp'].isoformat(),
            'checks': {
                name: {
                    'status': data['status'],
                    'message': data['message'],
                    'last_check': data['last_check'].isoformat() if data['last_check'] else None
                }
                for name, data in self.health_status['checks'].items()
            }
        }
    
    def update_metrics(self, metrics: Dict):
        """
        Update dashboard metrics
        
        Args:
            metrics: Dictionary of metrics
        """
        self.metrics_cache.update(metrics)
        self.last_update = datetime.now()
    
    def get_dashboard_data(self) -> Dict:
        """Get all dashboard data"""
        return {
            'health': self.get_health_status(),
            'metrics': self.metrics_cache,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_system_overview(self) -> Dict:
        """Get system overview for dashboard"""
        return {
            'uptime_seconds': (datetime.now() - self.health_status['timestamp']).total_seconds(),
            'health_status': self.health_status['status'],
            'active_checks': len(self.health_status['checks']),
            'metrics_count': len(self.metrics_cache)
        }


class MetricsDashboard:
    """
    Metrics aggregation and visualization for dashboard
    """
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics dashboard
        
        Args:
            retention_hours: Hours to retain metrics data
        """
        self.retention_hours = retention_hours
        self.metrics_series: Dict[str, List[Dict]] = {}
        self.aggregated_metrics: Dict[str, Any] = {}
        
    def add_metric_point(self, metric_name: str, value: float, 
                        tags: Dict[str, str] = None):
        """
        Add a metric data point
        
        Args:
            metric_name: Name of metric
            value: Metric value
            tags: Optional tags for filtering
        """
        if metric_name not in self.metrics_series:
            self.metrics_series[metric_name] = []
        
        self.metrics_series[metric_name].append({
            'timestamp': datetime.now(),
            'value': value,
            'tags': tags or {}
        })
        
        # Clean old data
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        self.metrics_series[metric_name] = [
            p for p in self.metrics_series[metric_name]
            if p['timestamp'] > cutoff
        ]
    
    def get_metric_series(self, metric_name: str, 
                         start_time: datetime = None,
                         end_time: datetime = None) -> List[Dict]:
        """
        Get metric time series
        
        Args:
            metric_name: Name of metric
            start_time: Start time filter
            end_time: End time filter
        
        Returns:
            List of metric points
        """
        if metric_name not in self.metrics_series:
            return []
        
        series = self.metrics_series[metric_name]
        
        if start_time:
            series = [p for p in series if p['timestamp'] >= start_time]
        if end_time:
            series = [p for p in series if p['timestamp'] <= end_time]
        
        return series
    
    def get_metric_statistics(self, metric_name: str, 
                             window_minutes: int = 60) -> Dict:
        """
        Get statistical summary of a metric
        
        Args:
            metric_name: Name of metric
            window_minutes: Time window for statistics
        
        Returns:
            Statistical summary
        """
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        series = self.get_metric_series(metric_name, start_time=cutoff)
        
        if not series:
            return {'error': 'No data available'}
        
        values = [p['value'] for p in series]
        
        return {
            'metric': metric_name,
            'window_minutes': window_minutes,
            'current': values[-1] if values else None,
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'std': (sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5,
            'count': len(values)
        }
    
    def get_top_metrics(self, limit: int = 10) -> List[Dict]:
        """
        Get top metrics by recent activity
        
        Args:
            limit: Number of metrics to return
        
        Returns:
            List of metric summaries
        """
        metrics_info = []
        
        for name, series in self.metrics_series.items():
            if series:
                recent = series[-1]['value']
                metrics_info.append({
                    'name': name,
                    'recent_value': recent,
                    'data_points': len(series)
                })
        
        metrics_info.sort(key=lambda x: x['data_points'], reverse=True)
        return metrics_info[:limit]
    
    def aggregate_by_time(self, metric_name: str, 
                          interval_minutes: int = 5,
                          agg_func: str = 'mean') -> List[Dict]:
        """
        Aggregate metric by time interval
        
        Args:
            metric_name: Name of metric
            interval_minutes: Aggregation interval
            agg_func: Aggregation function ('mean', 'sum', 'count')
        
        Returns:
            Aggregated data points
        """
        series = self.get_metric_series(metric_name)
        
        if not series:
            return []
        
        # Group by time bucket
        buckets = {}
        
        for point in series:
            bucket_time = point['timestamp'].replace(
                second=0, microsecond=0,
                minute=(point['timestamp'].minute // interval_minutes) * interval_minutes
            )
            
            if bucket_time not in buckets:
                buckets[bucket_time] = []
            buckets[bucket_time].append(point['value'])
        
        # Apply aggregation function
        result = []
        for bucket_time, values in sorted(buckets.items()):
            if agg_func == 'mean':
                agg_value = sum(values) / len(values)
            elif agg_func == 'sum':
                agg_value = sum(values)
            elif agg_func == 'count':
                agg_value = len(values)
            else:
                agg_value = values[-1]  # latest
            
            result.append({
                'timestamp': bucket_time,
                'value': agg_value
            })
        
        return result


class HealthCheckEndpoint:
    """
    FastAPI health check endpoint configuration
    """
    
    def __init__(self, app: FastAPI, dashboard_manager: DashboardManager):
        """
        Initialize health check endpoint
        
        Args:
            app: FastAPI application
            dashboard_manager: Dashboard manager instance
        """
        self.app = app
        self.dashboard = dashboard_manager
        self._setup_endpoints()
    
    def _setup_endpoints(self):
        """Setup health check endpoints"""
        
        @self.app.get("/health")
        async def health_check():
            """Basic health check"""
            status = self.dashboard.get_health_status()
            
            if status['status'] == 'healthy':
                return JSONResponse(content=status)
            else:
                return JSONResponse(content=status, status_code=503)
        
        @self.app.get("/health/live")
        async def liveness_check():
            """Liveness probe for Kubernetes"""
            return JSONResponse(content={'status': 'alive'})
        
        @self.app.get("/health/ready")
        async def readiness_check():
            """Readiness probe for Kubernetes"""
            status = self.dashboard.get_health_status()
            
            if status['status'] == 'healthy':
                return JSONResponse(content={'status': 'ready'})
            else:
                return JSONResponse(content={'status': 'not_ready'}, status_code=503)
        
        @self.app.get("/metrics/dashboard")
        async def metrics_dashboard():
            """Metrics dashboard endpoint"""
            data = self.dashboard.get_dashboard_data()
            return JSONResponse(content=data)
        
        @self.app.get("/metrics/{metric_name}")
        async def get_metric(metric_name: str, window_minutes: int = 60):
            """Get specific metric data"""
            from .dashboard import MetricsDashboard
            # This would need access to the metrics dashboard instance
            return JSONResponse(content={'message': 'Metric endpoint'})
    
    def create_html_dashboard(self) -> str:
        """Create HTML dashboard page"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Trading Bot Dashboard</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .status-card {
                    background-color: white;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .healthy { color: #27ae60; }
                .degraded { color: #f39c12; }
                .unhealthy { color: #e74c3c; }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .metric-card {
                    background-color: white;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    margin: 10px 0;
                }
                .timestamp {
                    color: #7f8c8d;
                    font-size: 12px;
                    margin-top: 10px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 AI Trading Bot Dashboard</h1>
                    <p>Real-time monitoring and metrics</p>
                </div>
                
                <div class="status-card">
                    <h2>System Health</h2>
                    <div id="health-status">Loading...</div>
                </div>
                
                <div class="metrics-grid" id="metrics-grid">
                    <!-- Metrics will be loaded here -->
                </div>
                
                <div class="timestamp" id="timestamp"></div>
            </div>
            
            <script>
                async function loadDashboard() {
                    try {
                        const response = await fetch('/health');
                        const health = await response.json();
                        
                        const healthDiv = document.getElementById('health-status');
                        healthDiv.innerHTML = `
                            <p>Status: <span class="${health.status}">${health.status}</span></p>
                            <p>Last Check: ${new Date(health.timestamp).toLocaleString()}</p>
                            <h3>Health Checks:</h3>
                            <ul>
                                ${Object.entries(health.checks).map(([name, check]) => `
                                    <li>
                                        <strong>${name}:</strong> 
                                        <span class="${check.status}">${check.status}</span>
                                        - ${check.message}
                                    </li>
                                `).join('')}
                            </ul>
                        `;
                        
                        const metricsResponse = await fetch('/metrics/dashboard');
                        const metrics = await metricsResponse.json();
                        
                        const metricsGrid = document.getElementById('metrics-grid');
                        if (metrics.metrics && Object.keys(metrics.metrics).length > 0) {
                            metricsGrid.innerHTML = Object.entries(metrics.metrics).map(([key, value]) => `
                                <div class="metric-card">
                                    <h3>${key.replace(/_/g, ' ').toUpperCase()}</h3>
                                    <div class="metric-value">${typeof value === 'number' ? value.toFixed(2) : value}</div>
                                </div>
                            `).join('');
                        } else {
                            metricsGrid.innerHTML = '<div class="metric-card">No metrics available</div>';
                        }
                        
                        document.getElementById('timestamp').innerHTML = `Last Updated: ${new Date().toLocaleString()}`;
                        
                    } catch (error) {
                        console.error('Failed to load dashboard:', error);
                        document.getElementById('health-status').innerHTML = '<p class="unhealthy">Failed to load health status</p>';
                    }
                }
                
                // Load dashboard on page load
                loadDashboard();
                
                // Auto-refresh every 30 seconds
                setInterval(loadDashboard, 30000);
            </script>
        </body>
        </html>
        """
        
        return html
    
    @property
    def dashboard_html(self) -> str:
        """Get HTML dashboard"""
        return self.create_html_dashboard()