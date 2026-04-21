"""
latency_sim.py
Part of the app/utils module.
Latency simulation for realistic backtesting and paper trading.
"""

import numpy as np
import random
import asyncio
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LatencyProfile(Enum):
    """Latency profiles for different scenarios"""
    IDEAL = "ideal"          # 0-1ms
    LOW = "low"              # 1-5ms
    NORMAL = "normal"        # 5-20ms
    HIGH = "high"            # 20-100ms
    EXTREME = "extreme"      # 100-500ms
    VOLATILE = "volatile"    # Varies widely
    SPIKY = "spiky"          # Mostly low but occasional spikes


@dataclass
class LatencyConfig:
    """Latency configuration"""
    base_latency_ms: float
    jitter_ms: float
    distribution: str  # 'normal', 'uniform', 'exponential'
    spike_probability: float = 0.0
    spike_multiplier: float = 5.0


class LatencySimulator:
    """
    Simulates network latency for realistic order execution.
    Supports multiple latency profiles and distributions.
    """
    
    def __init__(self, profile: LatencyProfile = LatencyProfile.NORMAL):
        """
        Initialize latency simulator
        
        Args:
            profile: Latency profile to use
        """
        self.profile = profile
        self.config = self._get_config_for_profile(profile)
        self.history: list = []
        
    def _get_config_for_profile(self, profile: LatencyProfile) -> LatencyConfig:
        """Get latency configuration for a profile"""
        profiles = {
            LatencyProfile.IDEAL: LatencyConfig(0.5, 0.2, 'normal', 0.0, 1.0),
            LatencyProfile.LOW: LatencyConfig(2, 1, 'normal', 0.02, 3),
            LatencyProfile.NORMAL: LatencyConfig(10, 5, 'normal', 0.05, 4),
            LatencyProfile.HIGH: LatencyConfig(50, 20, 'normal', 0.1, 3),
            LatencyProfile.EXTREME: LatencyConfig(200, 100, 'exponential', 0.2, 5),
            LatencyProfile.VOLATILE: LatencyConfig(20, 30, 'uniform', 0.15, 4),
            LatencyProfile.SPIKY: LatencyConfig(5, 3, 'normal', 0.3, 10),
        }
        return profiles.get(profile, profiles[LatencyProfile.NORMAL])
    
    def get_latency_ms(self) -> float:
        """
        Generate a latency value based on current profile
        
        Returns:
            Latency in milliseconds
        """
        # Check for spike
        if random.random() < self.config.spike_probability:
            latency = self.config.base_latency_ms * self.config.spike_multiplier
        else:
            # Generate based on distribution
            if self.config.distribution == 'normal':
                latency = np.random.normal(self.config.base_latency_ms, self.config.jitter_ms)
            elif self.config.distribution == 'uniform':
                latency = np.random.uniform(
                    max(0, self.config.base_latency_ms - self.config.jitter_ms),
                    self.config.base_latency_ms + self.config.jitter_ms
                )
            else:  # exponential
                latency = np.random.exponential(self.config.base_latency_ms)
        
        # Ensure non-negative
        latency = max(0, latency)
        
        # Record for statistics
        self.history.append(latency)
        if len(self.history) > 1000:
            self.history.pop(0)
        
        return latency
    
    async def simulate(self) -> float:
        """
        Simulate latency by sleeping for the generated duration
        
        Returns:
            Actual latency in milliseconds
        """
        latency_ms = self.get_latency_ms()
        await asyncio.sleep(latency_ms / 1000)
        return latency_ms
    
    def get_statistics(self) -> Dict:
        """
        Get latency statistics
        
        Returns:
            Statistics dictionary
        """
        if not self.history:
            return {'message': 'No latency data'}
        
        return {
            'profile': self.profile.value,
            'mean_ms': np.mean(self.history),
            'median_ms': np.median(self.history),
            'p95_ms': np.percentile(self.history, 95),
            'p99_ms': np.percentile(self.history, 99),
            'min_ms': np.min(self.history),
            'max_ms': np.max(self.history),
            'std_ms': np.std(self.history),
            'samples': len(self.history)
        }
    
    def change_profile(self, profile: LatencyProfile):
        """Change latency profile"""
        self.profile = profile
        self.config = self._get_config_for_profile(profile)
        logger.info(f"Latency profile changed to {profile.value}")
    
    def reset(self):
        """Reset latency history"""
        self.history.clear()
    
    def set_custom_config(self, base_latency_ms: float, jitter_ms: float,
                          distribution: str = 'normal', 
                          spike_probability: float = 0.0,
                          spike_multiplier: float = 5.0):
        """Set custom latency configuration"""
        self.config = LatencyConfig(
            base_latency_ms=base_latency_ms,
            jitter_ms=jitter_ms,
            distribution=distribution,
            spike_probability=spike_probability,
            spike_multiplier=spike_multiplier
        )
        self.profile = LatencyProfile.VOLATILE
        logger.info(f"Custom latency config set: base={base_latency_ms}ms, jitter={jitter_ms}ms")


class LatencyAwareExecutor:
    """
    Wrapper for order execution with simulated latency
    """
    
    def __init__(self, executor, latency_simulator: LatencySimulator = None):
        """
        Initialize latency-aware executor
        
        Args:
            executor: Original order executor
            latency_simulator: Latency simulator instance
        """
        self.executor = executor
        self.latency_sim = latency_simulator or LatencySimulator()
    
    async def execute(self, *args, **kwargs):
        """
        Execute order with simulated latency
        """
        # Simulate latency before execution
        latency_ms = await self.latency_sim.simulate()
        
        logger.debug(f"Execution latency: {latency_ms:.2f}ms")
        
        # Execute actual order
        result = await self.executor.execute(*args, **kwargs)
        
        # Add latency metadata
        if result and isinstance(result, dict):
            result['latency_ms'] = latency_ms
        
        return result