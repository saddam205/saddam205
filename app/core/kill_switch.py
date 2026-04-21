"""
kill_switch.py
Part of the app/core module.
Emergency kill switch for immediate trading halt.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class KillSwitchReason(Enum):
    """Reasons for kill switch activation"""
    MANUAL = "manual"
    HIGH_DRAWDOWN = "high_drawdown"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MARKET_CRASH = "market_crash"
    TECHNICAL_ISSUE = "technical_issue"
    DATA_CORRUPTION = "data_corruption"
    EXCHANGE_ERROR = "exchange_error"
    POSITION_LIMIT = "position_limit"
    VOLATILITY_SPIKE = "volatility_spike"
    UNKNOWN = "unknown"


class KillSwitch:
    """
    Emergency kill switch for immediate trading halt.
    Can be triggered manually or automatically based on conditions.
    """
    
    def __init__(self, state_file: str = "data/kill_switch_state.json"):
        """
        Initialize kill switch
        
        Args:
            state_file: File to persist kill switch state
        """
        self.state_file = state_file
        self.active = False
        self.activated_at = None
        self.activated_by = None
        self.reason = None
        self.details = {}
        self.history = []
        
        # Auto-recovery settings
        self.auto_recovery_enabled = True
        self.recovery_time_minutes = 30
        self.recovery_check_interval = 60  # seconds
        
        # Load previous state if exists
        self._load_state()
        
    def activate(self, reason: KillSwitchReason = KillSwitchReason.MANUAL, 
                 details: Optional[Dict] = None,
                 triggered_by: str = "system") -> Dict:
        """
        Activate kill switch - immediately halts all trading
        
        Args:
            reason: Reason for activation
            details: Additional details
            triggered_by: Who/what triggered the kill switch
        
        Returns:
            Activation confirmation
        """
        if self.active:
            logger.warning(f"Kill switch already active (activated at {self.activated_at})")
            return {
                'success': False,
                'message': 'Kill switch already active',
                'activated_at': self.activated_at
            }
        
        self.active = True
        self.activated_at = datetime.now()
        self.activated_by = triggered_by
        self.reason = reason
        self.details = details or {}
        
        # Log activation
        log_entry = {
            'timestamp': self.activated_at.isoformat(),
            'reason': reason.value,
            'triggered_by': triggered_by,
            'details': details
        }
        self.history.append(log_entry)
        
        # Save state
        self._save_state()
        
        logger.warning(f"🚨 KILL SWITCH ACTIVATED! Reason: {reason.value}")
        logger.warning(f"Triggered by: {triggered_by}")
        if details:
            logger.warning(f"Details: {details}")
        
        return {
            'success': True,
            'message': f'Kill switch activated: {reason.value}',
            'activated_at': self.activated_at.isoformat(),
            'reason': reason.value,
            'triggered_by': triggered_by
        }
    
    def deactivate(self, triggered_by: str = "system") -> Dict:
        """
        Deactivate kill switch - resumes trading
        
        Args:
            triggered_by: Who/what is deactivating
        
        Returns:
            Deactivation confirmation
        """
        if not self.active:
            logger.warning("Kill switch not active")
            return {
                'success': False,
                'message': 'Kill switch not active'
            }
        
        deactivated_at = datetime.now()
        
        # Log deactivation
        log_entry = {
            'timestamp': deactivated_at.isoformat(),
            'action': 'deactivated',
            'triggered_by': triggered_by,
            'duration_minutes': (deactivated_at - self.activated_at).total_seconds() / 60
        }
        self.history[-1]['deactivated_at'] = deactivated_at.isoformat()
        self.history[-1]['duration'] = log_entry['duration_minutes']
        
        self.active = False
        self.activated_at = None
        self.activated_by = None
        self.reason = None
        self.details = {}
        
        # Save state
        self._save_state()
        
        logger.info(f"✅ Kill switch deactivated by {triggered_by}")
        logger.info(f"Active for {log_entry['duration_minutes']:.1f} minutes")
        
        return {
            'success': True,
            'message': 'Kill switch deactivated',
            'deactivated_at': deactivated_at.isoformat(),
            'duration_minutes': log_entry['duration_minutes']
        }
    
    def is_active(self) -> bool:
        """Check if kill switch is active"""
        return self.active
    
    def get_status(self) -> Dict:
        """Get current kill switch status"""
        return {
            'active': self.active,
            'activated_at': self.activated_at.isoformat() if self.activated_at else None,
            'activated_by': self.activated_by,
            'reason': self.reason.value if self.reason else None,
            'details': self.details,
            'history_count': len(self.history),
            'last_activation': self.history[-1] if self.history else None
        }
    
    def check_auto_recovery(self) -> bool:
        """
        Check if auto-recovery conditions are met
        Returns True if kill switch should be deactivated
        """
        if not self.auto_recovery_enabled:
            return False
        
        if not self.active:
            return False
        
        if not self.activated_at:
            return False
        
        # Check if enough time has passed
        elapsed = (datetime.now() - self.activated_at).total_seconds() / 60
        if elapsed >= self.recovery_time_minutes:
            logger.info(f"Auto-recovery condition met after {elapsed:.1f} minutes")
            return True
        
        return False
    
    async def monitor(self):
        """Monitor kill switch status and handle auto-recovery"""
        while True:
            try:
                if self.active and self.auto_recovery_enabled:
                    if self.check_auto_recovery():
                        self.deactivate(triggered_by="auto_recovery")
                
                await asyncio.sleep(self.recovery_check_interval)
                
            except Exception as e:
                logger.error(f"Kill switch monitor error: {e}")
                await asyncio.sleep(30)
    
    def _save_state(self):
        """Persist kill switch state to file"""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            state = {
                'active': self.active,
                'activated_at': self.activated_at.isoformat() if self.activated_at else None,
                'activated_by': self.activated_by,
                'reason': self.reason.value if self.reason else None,
                'details': self.details,
                'history': self.history[-10:]  # Keep last 10 entries
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save kill switch state: {e}")
    
    def _load_state(self):
        """Load kill switch state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.active = state.get('active', False)
                
                # Only load if activation is recent (within last hour)
                if self.active and state.get('activated_at'):
                    activated_at = datetime.fromisoformat(state['activated_at'])
                    if (datetime.now() - activated_at).total_seconds() > 3600:
                        # Stale activation, clear it
                        self.active = False
                        logger.info("Cleared stale kill switch activation")
                    else:
                        self.activated_at = activated_at
                        self.activated_by = state.get('activated_by')
                        reason_str = state.get('reason')
                        if reason_str:
                            try:
                                self.reason = KillSwitchReason(reason_str)
                            except ValueError:
                                self.reason = KillSwitchReason.UNKNOWN
                        self.details = state.get('details', {})
                
                self.history = state.get('history', [])
                
        except Exception as e:
            logger.error(f"Failed to load kill switch state: {e}")
    
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get kill switch activation history"""
        return self.history[-limit:]
    
    def reset(self) -> Dict:
        """Reset kill switch and clear history"""
        self.active = False
        self.activated_at = None
        self.activated_by = None
        self.reason = None
        self.details = {}
        self.history = []
        self._save_state()
        
        logger.info("Kill switch reset")
        
        return {
            'success': True,
            'message': 'Kill switch reset successfully'
        }
    
    def set_auto_recovery(self, enabled: bool, recovery_minutes: int = None) -> Dict:
        """
        Configure auto-recovery settings
        
        Args:
            enabled: Enable/disable auto-recovery
            recovery_minutes: Minutes before auto-recovery (default: 30)
        
        Returns:
            Updated settings
        """
        self.auto_recovery_enabled = enabled
        
        if recovery_minutes:
            self.recovery_time_minutes = recovery_minutes
        
        logger.info(f"Auto-recovery {'enabled' if enabled else 'disabled'}")
        if enabled:
            logger.info(f"Recovery time: {self.recovery_time_minutes} minutes")
        
        return {
            'auto_recovery_enabled': self.auto_recovery_enabled,
            'recovery_time_minutes': self.recovery_time_minutes
        }