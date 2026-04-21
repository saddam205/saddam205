"""
stress_tests.py
Part of the app/backtesting module.
Stress testing for worst-case scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from .engine import BacktestEngine


@dataclass
class StressScenario:
    """Container for stress test scenario"""
    name: str
    description: str
    market_shock: float  # Percentage drop
    volatility_multiplier: float
    correlation_change: float
    liquidity_multiplier: float


class MarketCrashScenario:
    """Predefined market crash scenarios"""
    
    # 2008 Financial Crisis
    CRASH_2008 = StressScenario(
        name="2008 Financial Crisis",
        description="Global financial crisis with liquidity freeze",
        market_shock=-0.55,
        volatility_multiplier=3.0,
        correlation_change=0.8,
        liquidity_multiplier=0.1
    )
    
    # 2020 COVID Crash
    COVID_2020 = StressScenario(
        name="COVID-19 Crash",
        description="Pandemic-induced market panic",
        market_shock=-0.35,
        volatility_multiplier=2.5,
        correlation_change=0.7,
        liquidity_multiplier=0.3
    )
    
    # Flash Crash
    FLASH_CRASH = StressScenario(
        name="Flash Crash",
        description="Sudden liquidity event with rapid recovery",
        market_shock=-0.15,
        volatility_multiplier=4.0,
        correlation_change=0.9,
        liquidity_multiplier=0.05
    )
    
    # Crypto Winter
    CRYPTO_WINTER = StressScenario(
        name="Crypto Winter",
        description="Prolonged crypto bear market",
        market_shock=-0.70,
        volatility_multiplier=2.0,
        correlation_change=0.5,
        liquidity_multiplier=0.2
    )
    
    # Black Monday
    BLACK_MONDAY = StressScenario(
        name="Black Monday",
        description="1987-style crash with circuit breakers",
        market_shock=-0.23,
        volatility_multiplier=2.2,
        correlation_change=0.85,
        liquidity_multiplier=0.15
    )
    
    @classmethod
    def get_all_scenarios(cls) -> List[StressScenario]:
        """Get all predefined scenarios"""
        return [
            cls.CRASH_2008,
            cls.COVID_2020,
            cls.FLASH_CRASH,
            cls.CRYPTO_WINTER,
            cls.BLACK_MONDAY
        ]


class StressTester:
    """
    Stress testing framework for trading strategies
    Tests strategy resilience under extreme market conditions
    """
    
    def __init__(self, strategy_func, initial_capital: float = 100000):
        """
        Initialize stress tester
        
        Args:
            strategy_func: Strategy function to test
            initial_capital: Starting capital
        """
        self.strategy_func = strategy_func
        self.initial_capital = initial_capital
        self.results = {}
        
    def run_stress_test(self, data: pd.DataFrame, 
                        scenario: StressScenario) -> Dict:
        """
        Run stress test with given scenario
        
        Args:
            data: Historical data
            scenario: Stress scenario to apply
        
        Returns:
            Stress test results
        """
        # Apply stress to data
        stressed_data = self._apply_stress(data, scenario)
        
        # Run backtest on stressed data
        
        engine = BacktestEngine(initial_capital=self.initial_capital)
        metrics, trades, equity = engine.run_backtest(
            stressed_data, 
            self.strategy_func
        )
        
        # Calculate resilience metrics
        resilience = self._calculate_resilience(metrics, scenario)
        
        result = {
            'scenario': scenario.name,
            'description': scenario.description,
            'metrics': metrics,
            'resilience': resilience,
            'max_loss_percent': (self.initial_capital - metrics['final_capital']) / self.initial_capital * 100,
            'survived': metrics['final_capital'] > 0,
            'drawdown_exceeded': metrics['max_drawdown'] > 50,  # 50% drawdown threshold
            'trades_count': metrics['total_trades']
        }
        
        self.results[scenario.name] = result
        return result
    
    def _apply_stress(self, data: pd.DataFrame, scenario: StressScenario) -> pd.DataFrame:
        """Apply stress scenario to market data"""
        stressed = data.copy()
        
        # Apply market shock
        shock_start = len(data) // 2  # Apply shock at halfway point
        shock_duration = min(20, len(data) // 10)  # 10% of data or 20 days
        
        for i in range(shock_start, min(shock_start + shock_duration, len(data))):
            # Progressive shock
            shock_progress = (i - shock_start) / shock_duration
            current_shock = scenario.market_shock * shock_progress
            
            stressed.loc[stressed.index[i], 'Close'] = stressed.loc[stressed.index[i], 'Close'] * (1 + current_shock)
            
            # Adjust high/low
            stressed.loc[stressed.index[i], 'High'] = stressed.loc[stressed.index[i], 'High'] * (1 + current_shock * 0.5)
            stressed.loc[stressed.index[i], 'Low'] = stressed.loc[stressed.index[i], 'Low'] * (1 + current_shock)
        
        # Apply volatility multiplier to remaining data
        vol_start = shock_start + shock_duration
        for i in range(vol_start, len(data)):
            vol_mult = scenario.volatility_multiplier
            price_change = stressed.loc[stressed.index[i], 'Close'] / stressed.loc[stressed.index[i-1], 'Close'] - 1
            stressed.loc[stressed.index[i], 'Close'] = stressed.loc[stressed.index[i-1], 'Close'] * (1 + price_change * vol_mult)
        
        # Apply liquidity impact (slippage)
        stressed['slippage'] = stressed['Close'].pct_change() * (1 - scenario.liquidity_multiplier)
        
        return stressed
    
    def _calculate_resilience(self, metrics: Dict, scenario: StressScenario) -> Dict:
        """Calculate resilience metrics"""
        recovery_factor = metrics.get('recovery_factor', 0)
        max_drawdown = metrics.get('max_drawdown', 100)
        sharpe = metrics.get('sharpe_ratio', -1)
        
        # Calculate resilience score (0-100)
        score = 0
        if max_drawdown < 30:
            score += 40
        elif max_drawdown < 50:
            score += 20
        
        if recovery_factor > 1:
            score += 30
        elif recovery_factor > 0.5:
            score += 15
        
        if sharpe > 0:
            score += 30
        elif sharpe > -0.5:
            score += 10
        
        return {
            'resilience_score': score,
            'recovery_factor': recovery_factor,
            'max_drawdown_during_stress': max_drawdown,
            'sharpe_during_stress': sharpe,
            'rating': 'EXCELLENT' if score > 80 else 'GOOD' if score > 60 else 'FAIR' if score > 40 else 'POOR'
        }
    
    def run_comprehensive_stress_test(self, data: pd.DataFrame) -> Dict:
        """
        Run all predefined stress scenarios
        
        Args:
            data: Historical data
        
        Returns:
            Comprehensive stress test results
        """
        results = {}
        
        for scenario in MarketCrashScenario.get_all_scenarios():
            print(f"Running stress test: {scenario.name}...")
            result = self.run_stress_test(data, scenario)
            results[scenario.name] = result
        
        # Calculate overall resilience
        overall_score = np.mean([r['resilience']['resilience_score'] for r in results.values()])
        worst_drawdown = max([r['metrics']['max_drawdown'] for r in results.values()])
        worst_loss = min([r['max_loss_percent'] for r in results.values()])
        
        comprehensive = {
            'scenario_results': results,
            'overall_resilience_score': overall_score,
            'worst_drawdown': worst_drawdown,
            'worst_loss': worst_loss,
            'scenarios_survived': sum(1 for r in results.values() if r['survived']),
            'scenarios_failed': sum(1 for r in results.values() if not r['survived']),
            'overall_rating': 'PASS' if overall_score > 60 else 'FAIL'
        }
        
        return comprehensive
    
    def generate_stress_report(self, results: Dict) -> str:
        """Generate human-readable stress test report"""
        report = []
        report.append("=" * 60)
        report.append("STRESS TEST REPORT")
        report.append("=" * 60)
        
        for scenario_name, result in results.items():
            report.append(f"\n📊 {scenario_name}")
            report.append("-" * 40)
            report.append(f"  Description: {result['description']}")
            report.append(f"  Final Capital: ${result['metrics']['final_capital']:,.2f}")
            report.append(f"  Max Drawdown: {result['metrics']['max_drawdown']:.1f}%")
            report.append(f"  Max Loss: {result['max_loss_percent']:.1f}%")
            report.append(f"  Resilience Score: {result['resilience']['resilience_score']:.0f}/100")
            report.append(f"  Rating: {result['resilience']['rating']}")
            report.append(f"  Survived: {'✅ YES' if result['survived'] else '❌ NO'}")
        
        return "\n".join(report)