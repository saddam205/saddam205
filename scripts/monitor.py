#!/usr/bin/env python3
"""
monitor.py
System monitoring script for AI Trading Bot.
Monitors system health, performance metrics, and sends alerts.
"""

import psutil
import time
import logging
import json
import requests
from datetime import datetime
from pathlib import Path
import argparse
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SystemMonitor:
    """System monitoring class for trading bot"""
    
    def __init__(self, config_file: str = None):
        """
        Initialize system monitor
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.last_alert_time = {}
        self.metrics_history = []
        
    def _load_config(self, config_file: str) -> dict:
        """Load monitoring configuration"""
        default_config = {
            'api_endpoint': 'http://localhost:8000',
            'check_interval': 30,
            'cpu_threshold': 80,
            'memory_threshold': 85,
            'disk_threshold': 90,
            'latency_threshold': 1000,
            'alert_cooldown': 300,
            'telegram_bot_token': None,
            'telegram_chat_id': None,
            'email_enabled': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'alert_email': None,
            'email_password': None
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def check_system_health(self) -> dict:
        """Check system health metrics"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'load_avg': psutil.getloadavg(),
            'network_connections': len(psutil.net_connections()),
            'processes': len(psutil.pids())
        }
        
        # Add swap memory info
        swap = psutil.swap_memory()
        health['swap_percent'] = swap.percent
        
        return health
    
    def check_api_health(self) -> dict:
        """Check API endpoint health"""
        api_health = {
            'status': 'unknown',
            'response_time_ms': None,
            'error': None
        }
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.config['api_endpoint']}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            api_health['status'] = 'healthy' if response.status_code == 200 else 'degraded'
            api_health['response_time_ms'] = response_time
            api_health['status_code'] = response.status_code
            
            if response.status_code == 200:
                data = response.json()
                api_health['trading_mode'] = data.get('trading_mode')
                
        except requests.exceptions.Timeout:
            api_health['status'] = 'unhealthy'
            api_health['error'] = 'Timeout'
        except requests.exceptions.ConnectionError:
            api_health['status'] = 'unhealthy'
            api_health['error'] = 'Connection refused'
        except Exception as e:
            api_health['status'] = 'unhealthy'
            api_health['error'] = str(e)
        
        return api_health
    
    def check_trading_performance(self) -> dict:
        """Check trading performance metrics"""
        performance = {
            'status': 'healthy',
            'metrics': {}
        }
        
        try:
            response = requests.get(f"{self.config['api_endpoint']}/api/v1/performance", timeout=5)
            if response.status_code == 200:
                performance['metrics'] = response.json()
                
                # Check for concerning metrics
                if performance['metrics'].get('max_drawdown', 0) > 15:
                    performance['status'] = 'warning'
                    performance['alert'] = f"High drawdown: {performance['metrics']['max_drawdown']:.1f}%"
                elif performance['metrics'].get('win_rate', 100) < 40:
                    performance['status'] = 'warning'
                    performance['alert'] = f"Low win rate: {performance['metrics']['win_rate']:.1f}%"
                    
        except Exception as e:
            performance['status'] = 'unknown'
            performance['error'] = str(e)
        
        return performance
    
    def check_kill_switch(self) -> dict:
        """Check if kill switch is active"""
        try:
            response = requests.get(f"{self.config['api_endpoint']}/api/v1/system/kill-switch", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'active': data.get('active', False),
                    'reason': data.get('reason'),
                    'activated_at': data.get('activated_at')
                }
        except:
            pass
        return {'active': False}
    
    def should_alert(self, metric: str, value: float, threshold: float) -> bool:
        """Check if alert should be sent (with cooldown)"""
        current_time = time.time()
        last_alert = self.last_alert_time.get(metric, 0)
        
        if value >= threshold and (current_time - last_alert) > self.config['alert_cooldown']:
            self.last_alert_time[metric] = current_time
            return True
        return False
    
    def send_telegram_alert(self, message: str):
        """Send alert via Telegram"""
        if not self.config['telegram_bot_token'] or not self.config['telegram_chat_id']:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.config['telegram_bot_token']}/sendMessage"
            payload = {
                'chat_id': self.config['telegram_chat_id'],
                'text': f"🚨 *System Alert*\n\n{message}",
                'parse_mode': 'Markdown'
            }
            requests.post(url, json=payload, timeout=5)
            logger.info("Telegram alert sent")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
    def send_email_alert(self, subject: str, message: str):
        """Send alert via email"""
        if not self.config['email_enabled']:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['alert_email']
            msg['To'] = self.config['alert_email']
            msg['Subject'] = f"[AI Trading Bot] {subject}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['alert_email'], self.config['email_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info("Email alert sent")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def generate_report(self) -> str:
        """Generate monitoring report"""
        system_health = self.check_system_health()
        api_health = self.check_api_health()
        performance = self.check_trading_performance()
        kill_switch = self.check_kill_switch()
        
        report = f"""
{'='*60}
📊 SYSTEM MONITORING REPORT
{'='*60}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💻 SYSTEM HEALTH:
  CPU:        {system_health['cpu_percent']:.1f}%
  Memory:     {system_health['memory_percent']:.1f}%
  Disk:       {system_health['disk_usage']:.1f}%
  Load Avg:   {system_health['load_avg'][0]:.2f}, {system_health['load_avg'][1]:.2f}, {system_health['load_avg'][2]:.2f}

🌐 API STATUS:
  Status:     {api_health['status']}
  Response:   {api_health['response_time_ms']:.0f}ms
  Trading Mode: {api_health.get('trading_mode', 'N/A')}

📈 TRADING PERFORMANCE:
  Status:     {performance['status']}
  Total Trades: {performance['metrics'].get('total_trades', 0)}
  Win Rate:   {performance['metrics'].get('win_rate', 0):.1f}%
  Max Drawdown: {performance['metrics'].get('max_drawdown', 0):.1f}%

🛡️ KILL SWITCH:
  Active:     {'YES' if kill_switch['active'] else 'NO'}
  Reason:     {kill_switch.get('reason', 'N/A')}

{'='*60}
"""
        return report
    
    def run(self, one_shot: bool = False):
        """Run monitoring loop"""
        logger.info("Starting system monitor...")
        
        while True:
            try:
                # Check system health
                system_health = self.check_system_health()
                
                # CPU alert
                if self.should_alert('cpu', system_health['cpu_percent'], self.config['cpu_threshold']):
                    msg = f"⚠️ High CPU usage: {system_health['cpu_percent']:.1f}%"
                    logger.warning(msg)
                    self.send_telegram_alert(msg)
                
                # Memory alert
                if self.should_alert('memory', system_health['memory_percent'], self.config['memory_threshold']):
                    msg = f"⚠️ High memory usage: {system_health['memory_percent']:.1f}%"
                    logger.warning(msg)
                    self.send_telegram_alert(msg)
                
                # Disk alert
                if self.should_alert('disk', system_health['disk_usage'], self.config['disk_threshold']):
                    msg = f"⚠️ High disk usage: {system_health['disk_usage']:.1f}%"
                    logger.warning(msg)
                    self.send_telegram_alert(msg)
                
                # Check API health
                api_health = self.check_api_health()
                if api_health['status'] != 'healthy':
                    msg = f"⚠️ API health degraded: {api_health.get('error', 'Unknown error')}"
                    logger.warning(msg)
                    if self.should_alert('api', 1, 1):
                        self.send_telegram_alert(msg)
                
                # Check API response time
                if (api_health['response_time_ms'] and 
                    self.should_alert('latency', api_health['response_time_ms'], self.config['latency_threshold'])):
                    msg = f"⚠️ High API latency: {api_health['response_time_ms']:.0f}ms"
                    logger.warning(msg)
                    self.send_telegram_alert(msg)
                
                # Check trading performance
                performance = self.check_trading_performance()
                if performance['status'] == 'warning' and 'alert' in performance:
                    if self.should_alert('performance', 1, 1):
                        self.send_telegram_alert(f"⚠️ Trading alert: {performance['alert']}")
                
                # Check kill switch
                kill_switch = self.check_kill_switch()
                if kill_switch['active']:
                    msg = f"🚨 KILL SWITCH ACTIVE! Reason: {kill_switch.get('reason', 'Unknown')}"
                    logger.error(msg)
                    self.send_telegram_alert(msg)
                
                # Store metrics
                self.metrics_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'system': system_health,
                    'api': api_health
                })
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Log report periodically
                if len(self.metrics_history) % 20 == 0:
                    logger.info(self.generate_report())
                
                if one_shot:
                    print(self.generate_report())
                    break
                
                time.sleep(self.config['check_interval'])
                
            except KeyboardInterrupt:
                logger.info("Monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description='System Monitor for AI Trading Bot')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--one-shot', action='store_true', help='Run once and exit')
    parser.add_argument('--report', action='store_true', help='Generate and print report')
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(config_file=args.config)
    
    if args.report:
        print(monitor.generate_report())
    else:
        monitor.run(one_shot=args.one_shot)


if __name__ == "__main__":
    main()