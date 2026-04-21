"""
alerts.py
Part of the app/monitoring module.
Alert management system with multiple channels and severity levels.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels"""
    LOG = "log"
    TELEGRAM = "telegram"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"


@dataclass
class Alert:
    """Alert data structure"""
    title: str
    message: str
    severity: AlertSeverity
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'title': self.title,
            'message': self.message,
            'severity': self.severity.value,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'acknowledged': self.acknowledged
        }


class AlertManager:
    """
    Centralized alert management system with deduplication and escalation.
    """
    
    def __init__(self, deduplication_window: int = 300):
        """
        Initialize alert manager
        
        Args:
            deduplication_window: Time window for deduplication (seconds)
        """
        self.alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=10000)
        self.handlers: Dict[AlertChannel, Callable] = {}
        self.deduplication_window = deduplication_window
        self.last_alert_per_key: Dict[str, datetime] = {}
        
        # Register default handler
        self.register_handler(AlertChannel.LOG, self._log_handler)
        
    def register_handler(self, channel: AlertChannel, handler: Callable):
        """
        Register an alert handler
        
        Args:
            channel: Alert channel
            handler: Handler function
        """
        self.handlers[channel] = handler
        logger.info(f"Registered alert handler for {channel.value}")
    
    async def send_alert(self, title: str, message: str, 
                        severity: AlertSeverity = AlertSeverity.INFO,
                        source: str = "system",
                        channels: List[AlertChannel] = None,
                        metadata: Dict = None) -> Alert:
        """
        Send an alert
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            source: Alert source
            channels: List of channels to send to
            metadata: Additional metadata
        
        Returns:
            Alert object
        """
        # Check deduplication
        alert_key = f"{source}:{title}"
        if alert_key in self.last_alert_per_key:
            time_since_last = (datetime.now() - self.last_alert_per_key[alert_key]).total_seconds()
            if time_since_last < self.deduplication_window:
                logger.debug(f"Deduplicated alert: {title}")
                return None
        
        # Create alert
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            source=source,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        self.alert_history.append(alert)
        self.last_alert_per_key[alert_key] = datetime.now()
        
        # Keep only recent alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]
        
        # Send to channels
        channels = channels or [AlertChannel.LOG]
        
        for channel in channels:
            if channel in self.handlers:
                try:
                    await self.handlers[channel](alert)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel.value}: {e}")
        
        # Log critical alerts
        if severity == AlertSeverity.CRITICAL:
            logger.critical(f"CRITICAL ALERT: {title} - {message}")
        elif severity == AlertSeverity.ERROR:
            logger.error(f"ALERT: {title} - {message}")
        elif severity == AlertSeverity.WARNING:
            logger.warning(f"Alert: {title} - {message}")
        
        return alert
    
    async def _log_handler(self, alert: Alert):
        """Log alert handler"""
        # Already logged in send_alert, just pass
        pass
    
    async def acknowledge_alert(self, alert_index: int, acknowledged_by: str):
        """
        Acknowledge an alert
        
        Args:
            alert_index: Index of alert in list
            acknowledged_by: Person/system acknowledging
        """
        if 0 <= alert_index < len(self.alerts):
            alert = self.alerts[alert_index]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            logger.info(f"Alert acknowledged by {acknowledged_by}: {alert.title}")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active (unacknowledged) alerts"""
        active = [a for a in self.alerts if not a.acknowledged]
        
        if severity:
            active = [a for a in active if a.severity == severity]
        
        return active
    
    def get_alert_summary(self) -> Dict:
        """Get alert summary statistics"""
        active_alerts = self.get_active_alerts()
        
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'critical_active': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'error_active': len([a for a in active_alerts if a.severity == AlertSeverity.ERROR]),
            'warning_active': len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
            'last_24h': len([a for a in self.alert_history if a.timestamp > datetime.now() - timedelta(hours=24)])
        }
    
    def clear_alerts(self, severity: Optional[AlertSeverity] = None):
        """Clear alerts"""
        if severity:
            self.alerts = [a for a in self.alerts if a.severity != severity]
        else:
            self.alerts = []
        logger.info(f"Cleared alerts (severity={severity})")


class TelegramAlertHandler:
    """Telegram alert handler"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram handler
        
        Args:
            bot_token: Telegram bot token
            chat_id: Chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    async def send(self, alert: Alert):
        """Send alert via Telegram"""
        import aiohttp
        
        # Format message
        emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }.get(alert.severity, "📢")
        
        message = f"{emoji} *{alert.title}*\n\n{alert.message}\n\n*Source:* {alert.source}\n*Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send via HTTP
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Failed to send Telegram alert: {await response.text()}")


class EmailAlertHandler:
    """Email alert handler"""
    
    def __init__(self, smtp_server: str, from_email: str, to_emails: List[str]):
        """
        Initialize email handler
        
        Args:
            smtp_server: SMTP server
            from_email: From email address
            to_emails: List of recipient emails
        """
        self.smtp_server = smtp_server
        self.from_email = from_email
        self.to_emails = to_emails
    
    async def send(self, alert: Alert):
        """Send alert via email"""
        import smtplib
        from email.mime.text import MIMEText
        
        # Only send critical alerts via email
        if alert.severity not in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            return
        
        subject = f"[{alert.severity.value.upper()}] {alert.title}"
        body = f"""
        Alert from {alert.source}
        Time: {alert.timestamp}
        
        {alert.message}
        
        Metadata: {alert.metadata}
        """
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)
        
        # Send email (simplified - would need SMTP connection)
        logger.info(f"Email alert would be sent: {subject}")