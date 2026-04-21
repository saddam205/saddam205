"""
walk_forward_validator.py
Part of the app/validation module.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WalkForwardValidator:
    """
    Professional walk-forward testing to prevent overfitting
    No peeking into future data!
    """
    
    def __init__(self, model_class, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        self.model_class = model_class
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.results = []
        
    def run_validation(self, data, n_splits=5):
        """
        Run walk-forward validation with multiple splits
        
        Args:
            data: Full dataset with datetime index
            n_splits: Number of walk-forward windows
        """
        print("="*60)
        print("🔬 WALK-FORWARD VALIDATION")
        print("="*60)
        print(f"Total Data Points: {len(data)}")
        print(f"Number of Splits: {n_splits}")
        print(f"Train:Val:Test = {self.train_ratio:.0%}:{self.val_ratio:.0%}:{self.test_ratio:.0%}")
        print("="*60)
        
        # Calculate window sizes
        total_points = len(data)
        train_size = int(total_points * self.train_ratio)
        val_size = int(total_points * self.val_ratio)
        test_size = int(total_points * self.test_ratio)
        
        # Create rolling windows
        results = []
        
        for split in range(n_splits):
            print(f"\n📊 WALK-FORWARD WINDOW {split + 1}/{n_splits}")
            print("-"*40)
            
            # Define time windows (no overlap, no peeking)
            train_end = train_size + (split * test_size)
            val_end = train_end + val_size
            test_end = val_end + test_size
            
            # Ensure we don't exceed data
            if test_end > len(data):
                print("⚠️ Reached end of data. Stopping.")
                break
            
            # Split data chronologically
            train_data = data.iloc[:train_end]
            val_data = data.iloc[train_end:val_end]
            test_data = data.iloc[val_end:test_end]
            
            print(f"Training Period: {train_data.index[0]} to {train_data.index[-1]}")
            print(f"Validation Period: {val_data.index[0]} to {val_data.index[-1]}")
            print(f"Test Period: {test_data.index[0]} to {test_data.index[-1]}")
            
            # Train model
            model = self.model_class()
            model.fit(train_data, val_data)
            
            # Test on out-of-sample data
            test_results = model.evaluate(test_data)
            
            # Record results
            window_result = {
                'split': split + 1,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'val_start': val_data.index[0],
                'val_end': val_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'test_accuracy': test_results.get('accuracy', 0),
                'test_sharpe': test_results.get('sharpe_ratio', 0),
                'test_max_dd': test_results.get('max_drawdown', 0),
                'test_win_rate': test_results.get('win_rate', 0),
                'test_trades': test_results.get('total_trades', 0)
            }
            
            results.append(window_result)
            
            # Print results
            print(f"\n📈 Test Results:")
            print(f"  Accuracy: {window_result['test_accuracy']:.2%}")
            print(f"  Sharpe Ratio: {window_result['test_sharpe']:.2f}")
            print(f"  Max Drawdown: {window_result['test_max_dd']:.2%}")
            print(f"  Win Rate: {window_result['test_win_rate']:.2%}")
            print(f"  Trades: {window_result['test_trades']}")
        
        # Aggregate results
        self.results = results
        self._calculate_aggregate_metrics()
        
        return results
    
    def _calculate_aggregate_metrics(self):
        """Calculate aggregate performance across all windows"""
        if not self.results:
            return
        
        accuracies = [r['test_accuracy'] for r in self.results]
        sharpes = [r['test_sharpe'] for r in self.results]
        drawdowns = [r['test_max_dd'] for r in self.results]
        win_rates = [r['test_win_rate'] for r in self.results]
        
        self.aggregate = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'mean_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'mean_max_dd': np.mean(drawdowns),
            'max_max_dd': np.max(drawdowns),
            'mean_win_rate': np.mean(win_rates),
            'robustness_score': self._calculate_robustness_score()
        }
        
        print("\n" + "="*60)
        print("📊 AGGREGATE WALK-FORWARD RESULTS")
        print("="*60)
        print(f"Mean Accuracy: {self.aggregate['mean_accuracy']:.2%} ± {self.aggregate['std_accuracy']:.2%}")
        print(f"Accuracy Range: {self.aggregate['min_accuracy']:.2%} - {self.aggregate['max_accuracy']:.2%}")
        print(f"Mean Sharpe Ratio: {self.aggregate['mean_sharpe']:.2f} ± {self.aggregate['std_sharpe']:.2f}")
        print(f"Mean Max Drawdown: {self.aggregate['mean_max_dd']:.2%}")
        print(f"Worst Drawdown: {self.aggregate['max_max_dd']:.2%}")
        print(f"Mean Win Rate: {self.aggregate['mean_win_rate']:.2%}")
        print(f"\nRobustness Score: {self.aggregate['robustness_score']:.2f}/1.00")
        
        # Final verdict
        if self.aggregate['robustness_score'] > 0.7:
            print("\n✅ SYSTEM IS ROBUST - Passes walk-forward validation")
        elif self.aggregate['robustness_score'] > 0.5:
            print("\n⚠️ SYSTEM IS MODERATELY ROBUST - Use with caution")
        else:
            print("\n❌ SYSTEM IS OVERFITTED - Needs improvement")
    
    def _calculate_robustness_score(self):
        """Calculate how robust the system is across different periods"""
        # Penalize high variance in accuracy
        accuracy_stability = 1 - min(self.aggregate['std_accuracy'] / 0.1, 1)
        
        # Reward positive Sharpe ratio
        sharpe_score = min(self.aggregate['mean_sharpe'] / 2, 1)
        
        # Penalize high drawdown
        drawdown_score = 1 - min(self.aggregate['mean_max_dd'] / 0.2, 1)
        
        # Combined score
        robustness = (accuracy_stability * 0.4 + 
                     sharpe_score * 0.4 + 
                     drawdown_score * 0.2)
        
        return robustness

# Example usage
class SimpleModel:
    """Placeholder for your actual model"""
    def fit(self, train_data, val_data):
        # Train your model here
        pass
    
    def evaluate(self, test_data):
        # Evaluate and return metrics
        return {
            'accuracy': 0.58,
            'sharpe_ratio': 1.6,
            'max_drawdown': 0.08,
            'win_rate': 0.62,
            'total_trades': 45
        }

# Run walk-forward
validator = WalkForwardValidator(SimpleModel)
results = validator.run_validation(data, n_splits=5)
class RealisticCostCalculator:
    """
    Professional cost modeling for realistic P&L
    """
    
    def __init__(self, 
                 maker_fee=0.0005,      # 0.05% maker fee
                 taker_fee=0.001,        # 0.1% taker fee
                 slippage_model='adaptive',
                 spread_model='volatility_based'):
        
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_model = slippage_model
        self.spread_model = spread_model
        
        # Historical slippage data
        self.slippage_history = []
        
    def calculate_trade_cost(self, 
                            trade_size: float,
                            price: float,
                            volatility: float,
                            volume: float,
                            is_market_order: bool = True) -> dict:
        """
        Calculate realistic trade costs including:
        - Exchange fees
        - Slippage
        - Spread
        - Market impact
        """
        
        # 1. Exchange Fee
        fee_rate = self.taker_fee if is_market_order else self.maker_fee
        exchange_fee = trade_size * price * fee_rate
        
        # 2. Slippage (price movement during execution)
        slippage = self._calculate_slippage(trade_size, volume, volatility, is_market_order)
        slippage_cost = trade_size * price * slippage
        
        # 3. Spread Cost
        spread = self._calculate_spread(volatility, volume)
        spread_cost = trade_size * price * (spread / 2)  # Half spread on entry
        
        # 4. Market Impact (large orders)
        market_impact = self._calculate_market_impact(trade_size, volume, price)
        
        total_cost = exchange_fee + slippage_cost + spread_cost + market_impact
        total_cost_pct = total_cost / (trade_size * price) * 100
        
        return {
            'exchange_fee': exchange_fee,
            'exchange_fee_pct': fee_rate * 100,
            'slippage': slippage,
            'slippage_cost': slippage_cost,
            'slippage_pct': slippage * 100,
            'spread': spread,
            'spread_cost': spread_cost,
            'spread_pct': (spread / 2) * 100,
            'market_impact': market_impact,
            'market_impact_pct': (market_impact / (trade_size * price)) * 100,
            'total_cost': total_cost,
            'total_cost_pct': total_cost_pct
        }
    
    def _calculate_slippage(self, trade_size, volume, volatility, is_market_order):
        """
        Calculate expected slippage based on:
        - Trade size relative to volume
        - Current volatility
        - Order type
        """
        # Base slippage
        if is_market_order:
            base_slippage = 0.0005  # 0.05% base for market orders
        else:
            base_slippage = 0.0001  # 0.01% for limit orders
        
        # Size impact (larger orders = more slippage)
        size_ratio = trade_size / volume if volume > 0 else 0.01
        size_impact = min(size_ratio * 10, 0.005)  # Max 0.5%
        
        # Volatility impact
        vol_impact = volatility * 0.5  # 50% of volatility as slippage
        
        # Total slippage
        total_slippage = base_slippage + size_impact + vol_impact
        
        # Record for adaptive model
        self.slippage_history.append({
            'trade_size': trade_size,
            'volume': volume,
            'volatility': volatility,
            'slippage': total_slippage
        })
        
        return min(total_slippage, 0.01)  # Cap at 1%
    
    def _calculate_spread(self, volatility, volume):
        """
        Calculate bid-ask spread based on market conditions
        """
        # Base spread (0.05% for liquid markets)
        base_spread = 0.0005
        
        # Volatility adjustment
        vol_adjustment = volatility * 0.5
        
        # Liquidity adjustment
        liquidity_score = min(volume / 1000000, 1)  # Normalized by $1M volume
        liquidity_adjustment = (1 - liquidity_score) * 0.001
        
        total_spread = base_spread + vol_adjustment + liquidity_adjustment
        
        return min(total_spread, 0.005)  # Cap at 0.5%
    
    def _calculate_market_impact(self, trade_size, volume, price):
        """
        Calculate market impact for large orders (Almgren-Chriss model)
        """
        if volume == 0:
            return 0
        
        # Participation rate (trade size / volume)
        participation = trade_size / volume
        
        # Impact model: impact = η * σ * (participation)^0.5
        eta = 0.1  # Impact coefficient
        sigma = 0.02  # Daily volatility assumption
        
        impact_pct = eta * sigma * np.sqrt(participation)
        
        # Cap impact
        impact_pct = min(impact_pct, 0.01)  # Max 1%
        
        return trade_size * price * impact_pct

class CostAdjustedBacktest:
    """
    Backtest with realistic costs
    """
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.cost_calculator = RealisticCostCalculator()
        self.trades = []
        
    def execute_trade(self, side, size, price, timestamp, volatility, volume):
        """Execute trade with realistic costs"""
        
        # Calculate costs
        costs = self.cost_calculator.calculate_trade_cost(
            trade_size=size,
            price=price,
            volatility=volatility,
            volume=volume,
            is_market_order=True
        )
        
        # Apply costs to P&L
        if side == 'BUY':
            total_cost = size * price + costs['total_cost']
            self.capital -= total_cost
            
        else:  # SELL
            net_proceeds = size * price - costs['total_cost']
            self.capital += net_proceeds
        
        # Record trade with costs
        trade_record = {
            'timestamp': timestamp,
            'side': side,
            'size': size,
            'price': price,
            'costs': costs,
            'net_execution': total_cost if side == 'BUY' else net_proceeds,
            'balance_after': self.capital
        }
        
        self.trades.append(trade_record)
        
        return trade_record
    
    def get_realistic_performance(self):
        """Get performance metrics with costs included"""
        if not self.trades:
            return {}
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        total_costs = sum(t['costs']['total_cost'] for t in self.trades)
        cost_drag = total_costs / self.initial_capital * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_costs': total_costs,
            'cost_drag_pct': cost_drag,
            'cost_impact_on_return': total_return - (total_return + cost_drag),
            'total_trades': len(self.trades),
            'avg_cost_per_trade': total_costs / len(self.trades) if self.trades else 0
        }



class MarketStressTester:
    """
    Comprehensive stress testing for extreme scenarios
    """
    
    def __init__(self, strategy):
        self.strategy = strategy
        self.scenarios = {}
        self.results = {}
        
    def create_stress_scenarios(self):
        """Create realistic stress test scenarios"""
        
        scenarios = {
            'flash_crash': {
                'name': 'Flash Crash (May 2021 Style)',
                'description': '50% drop in 24 hours',
                'price_path': self._generate_flash_crash(),
                'volatility_multiplier': 5,
                'volume_multiplier': 3,
                'correlation_break': True
            },
            'bear_market': {
                'name': 'Prolonged Bear Market',
                'description': '80% drawdown over 12 months',
                'price_path': self._generate_bear_market(),
                'volatility_multiplier': 2,
                'volume_multiplier': 0.5,
                'correlation_break': False
            },
            'liquidity_crisis': {
                'name': 'Liquidity Crisis',
                'description': 'Bid-ask spreads widen 10x',
                'price_path': self._generate_normal(),
                'volatility_multiplier': 3,
                'volume_multiplier': 0.2,
                'spread_multiplier': 10,
                'correlation_break': True
            },
            'high_volatility': {
                'name': 'High Volatility Regime',
                'description': 'VIX at 80+',
                'price_path': self._generate_high_vol(),
                'volatility_multiplier': 4,
                'volume_multiplier': 1.5,
                'correlation_break': False
            },
            'black_swan': {
                'name': 'Black Swan Event',
                'description': 'Unprecedented market event (COVID/FTX style)',
                'price_path': self._generate_black_swan(),
                'volatility_multiplier': 8,
                'volume_multiplier': 2,
                'liquidity_evaporates': True,
                'correlation_break': True
            },
            'sideways_market': {
                'name': 'Sideways/Choppy Market',
                'description': 'No trend, high noise',
                'price_path': self._generate_sideways(),
                'volatility_multiplier': 1.2,
                'volume_multiplier': 0.8,
                'trend_strength': 0.1,
                'correlation_break': False
            }
        }
        
        return scenarios
    
    def _generate_flash_crash(self):
        """Generate flash crash price path"""
        days = 30
        prices = [100]
        
        for i in range(days):
            if i < 1:  # Crash day
                prices.append(prices[-1] * 0.7)  # 30% drop
            elif i < 3:  # Recovery
                prices.append(prices[-1] * 1.1)  # 10% bounce
            else:
                prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
        
        return prices
    
    def _generate_bear_market(self):
        """Generate prolonged bear market"""
        days = 365
        prices = [100]
        
        for i in range(days):
            # 60% chance of down day
            if np.random.random() < 0.6:
                change = np.random.normal(-0.003, 0.015)
            else:
                change = np.random.normal(0.001, 0.015)
            
            prices.append(prices[-1] * (1 + change))
        
        return prices
    
    def _generate_high_vol(self):
        """Generate high volatility regime"""
        days = 90
        prices = [100]
        
        for i in range(days):
            change = np.random.normal(0, 0.05)  # 5% daily vol
            prices.append(prices[-1] * (1 + change))
        
        return prices
    
    def _generate_black_swan(self):
        """Generate black swan event"""
        days = 60
        prices = [100]
        
        for i in range(days):
            if i < 2:  # Crash
                prices.append(prices[-1] * 0.5)  # 50% drop
            elif i < 5:  # Continued panic
                prices.append(prices[-1] * 0.85)  # 15% more
            elif i < 10:  # Bottom
                prices.append(prices[-1] * 0.95)
            else:  # Slow recovery
                change = np.random.normal(0.005, 0.025)
                prices.append(prices[-1] * (1 + change))
        
        return prices
    
    def _generate_sideways(self):
        """Generate sideways/choppy market"""
        days = 180
        prices = [100]
        
        for i in range(days):
            # Low drift, high noise
            change = np.random.normal(0, 0.02)  # 2% daily noise
            prices.append(prices[-1] * (1 + change))
        
        # Normalize to end near start
        final_factor = 100 / prices[-1]
        prices = [p * final_factor for p in prices]
        
        return prices
    
    def _generate_normal(self):
        """Generate normal market conditions"""
        days = 180
        prices = [100]
        
        for i in range(days):
            change = np.random.normal(0.0005, 0.015)  # 0.05% drift
            prices.append(prices[-1] * (1 + change))
        
        return prices
    
    def run_stress_tests(self):
        """Execute all stress tests"""
        print("="*60)
        print("🌪️ MARKET STRESS TESTING")
        print("="*60)
        
        scenarios = self.create_stress_scenarios()
        results = {}
        
        for name, scenario in scenarios.items():
            print(f"\n📊 Testing: {scenario['name']}")
            print(f"   {scenario['description']}")
            
            # Run strategy on stress scenario
            test_results = self._run_scenario_test(scenario)
            
            results[name] = {
                'scenario': scenario,
                'results': test_results,
                'passed': test_results['max_drawdown'] < 0.30,  # <30% drawdown
                'survived': test_results['final_capital'] > 0
            }
            
            # Print results
            print(f"   Final Capital: ${test_results['final_capital']:,.2f}")
            print(f"   Total Return: {test_results['total_return']:.2f}%")
            print(f"   Max Drawdown: {test_results['max_drawdown']:.2%}")
            print(f"   Sharpe Ratio: {test_results['sharpe_ratio']:.2f}")
            print(f"   Status: {'✅ PASSED' if results[name]['passed'] else '❌ FAILED'}")
        
        # Summary
        self._print_stress_summary(results)
        
        return results
    
    def _run_scenario_test(self, scenario):
        """Run strategy on specific scenario"""
        price_path = scenario['price_path']
        
        # Initialize tracking
        capital = 100000
        position = 0
        equity_curve = []
        trades = []
        
        for i, price in enumerate(price_path):
            # Apply volatility multiplier
            vol = scenario['volatility_multiplier'] * 0.02
            
            # Get strategy signal
            signal = self.strategy.get_signal(price, vol)
            
            # Execute trades
            if signal == 'BUY' and position == 0:
                position = capital * 0.2 / price  # 20% position
                capital -= position * price
                
            elif signal == 'SELL' and position > 0:
                capital += position * price
                pnl = (price - trades[-1]['price']) * position if trades else 0
                position = 0
            
            # Record equity
            equity = capital + (position * price)
            equity_curve.append(equity)
        
        # Calculate metrics
        returns = np.diff(equity_curve) / equity_curve[:-1]
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        return {
            'final_capital': equity_curve[-1],
            'total_return': (equity_curve[-1] - 100000) / 100000 * 100,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 0.0001) * np.sqrt(252),
            'equity_curve': equity_curve
        }
    
    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _print_stress_summary(self, results):
        """Print stress test summary"""
        print("\n" + "="*60)
        print("📊 STRESS TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in results.values() if r['passed'])
        total = len(results)
        
        print(f"\nScenarios Passed: {passed}/{total} ({passed/total*100:.0f}%)")
        print(f"Scenarios Failed: {total-passed}/{total}")
        
        if passed == total:
            print("\n✅ SYSTEM PASSED ALL STRESS TESTS!")
            print("   Robust to market crashes, high volatility, and liquidity crises")
        elif passed >= total * 0.7:
            print("\n⚠️ SYSTEM PASSED MOST STRESS TESTS")
            print("   Consider adding crash protection mechanisms")
        else:
            print("\n❌ SYSTEM FAILED MULTIPLE STRESS TESTS")
            print("   Significant improvements needed for crash protection")
        
        # Show worst-case scenario
        worst_scenario = min(results.items(), key=lambda x: x[1]['results']['max_drawdown'])
        print(f"\n🔴 WORST-CASE SCENARIO: {worst_scenario[0]}")
        print(f"   Max Drawdown: {worst_scenario[1]['results']['max_drawdown']:.2%}")
        
        
        class EquityCurveAnalyzer:
    """
    Professional equity curve analysis
    Not just accuracy - but consistency, drawdown, and quality
    """
    
    def __init__(self, equity_curve, trades):
        self.equity_curve = np.array(equity_curve)
        self.trades = trades
        self.metrics = {}
        
    def analyze(self):
        """Comprehensive equity curve analysis"""
        print("="*60)
        print("📈 EQUITY CURVE ANALYSIS")
        print("="*60)
        
        # Calculate all metrics
        self._calculate_returns()
        self._calculate_drawdowns()
        self._calculate_consistency()
        self._calculate_quality_metrics()
        
        # Print results
        self._print_analysis()
        
        return self.metrics
    
    def _calculate_returns(self):
        """Calculate return metrics"""
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        self.metrics['returns'] = {
            'total_return': (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0] * 100,
            'cagr': self._calculate_cagr(),
            'avg_daily_return': np.mean(returns) * 100,
            'std_daily_return': np.std(returns) * 100,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 0.0001) * np.sqrt(252),
            'sortino_ratio': self._calculate_sortino(returns),
            'calmar_ratio': self._calculate_calmar()
        }
    
    def _calculate_drawdowns(self):
        """Calculate drawdown metrics"""
        peak = self.equity_curve[0]
        drawdowns = []
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            drawdowns.append(dd)
        
        self.metrics['drawdowns'] = {
            'max_drawdown': np.max(drawdowns) * 100,
            'avg_drawdown': np.mean(drawdowns) * 100,
            'drawdown_duration': self._calculate_drawdown_duration(drawdowns),
            'recovery_factor': self.metrics['returns']['total_return'] / (np.max(drawdowns) * 100) if np.max(drawdowns) > 0 else float('inf'),
            'underwater_periods': self._detect_underwater_periods(drawdowns)
        }
    
    def _calculate_consistency(self):
        """Calculate consistency metrics"""
        if not self.trades:
            return
        
        # Rolling metrics
        window = min(20, len(self.trades))
        rolling_returns = []
        
        for i in range(len(self.trades) - window):
            window_trades = self.trades[i:i+window]
            window_pnl = sum(t.get('pnl', 0) for t in window_trades)
            rolling_returns.append(window_pnl)
        
        self.metrics['consistency'] = {
            'win_rate': sum(1 for t in self.trades if t.get('pnl', 0) > 0) / len(self.trades) * 100,
            'avg_win': np.mean([t['pnl'] for t in self.trades if t.get('pnl', 0) > 0]) if any(t.get('pnl', 0) > 0 for t in self.trades) else 0,
            'avg_loss': np.mean([t['pnl'] for t in self.trades if t.get('pnl', 0) < 0]) if any(t.get('pnl', 0) < 0 for t in self.trades) else 0,
            'profit_factor': abs(sum(t['pnl'] for t in self.trades if t['pnl'] > 0) / 
                                 sum(t['pnl'] for t in self.trades if t['pnl'] < 0)) if any(t['pnl'] < 0 for t in self.trades) else float('inf'),
            'max_consecutive_wins': self._max_consecutive('win'),
            'max_consecutive_losses': self._max_consecutive('loss'),
            'rolling_returns_std': np.std(rolling_returns) if rolling_returns else 0,
            'monthly_consistency': self._calculate_monthly_consistency()
        }
    
    def _calculate_quality_metrics(self):
        """Calculate advanced quality metrics"""
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        self.metrics['quality'] = {
            'kelly_criterion': self._calculate_kelly(),
            'tail_ratio': self._calculate_tail_ratio(returns),
            'gain_to_pain_ratio': self.metrics['returns']['total_return'] / (self.metrics['drawdowns']['max_drawdown']),
            'ulcer_index': self._calculate_ulcer_index(),
            'sterling_ratio': self._calculate_sterling_ratio(),
            'burke_ratio': self._calculate_burke_ratio(returns)
        }
    
    def _calculate_cagr(self):
        """Calculate Compound Annual Growth Rate"""
        days = len(self.equity_curve)
        years = days / 365
        total_return = self.equity_curve[-1] / self.equity_curve[0]
        return (total_return ** (1 / years) - 1) * 100 if years > 0 else 0
    
    def _calculate_sortino(self, returns):
        """Calculate Sortino Ratio (downside risk only)"""
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0.0001
        return np.mean(returns) / downside_std * np.sqrt(252)
    
    def _calculate_calmar(self):
        """Calculate Calmar Ratio (return / max drawdown)"""
        max_dd = self.metrics['drawdowns']['max_drawdown'] / 100 if 'drawdowns' in self.metrics else 0.01
        return self.metrics['returns']['cagr'] / max_dd if max_dd > 0 else 0
    
    def _calculate_drawdown_duration(self, drawdowns):
        """Calculate average drawdown duration"""
        in_dd = False
        durations = []
        current_duration = 0
        
        for dd in drawdowns:
            if dd > 0.01:  # >1% drawdown
                if not in_dd:
                    in_dd = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_dd:
                    durations.append(current_duration)
                    in_dd = False
                    current_duration = 0
        
        return np.mean(durations) if durations else 0
    
    def _detect_underwater_periods(self, drawdowns):
        """Detect and count underwater periods"""
        underwater = 0
        in_water = False
        
        for dd in drawdowns:
            if dd > 0.05 and not in_water:  # >5% drawdown
                in_water = True
                underwater += 1
            elif dd < 0.01:
                in_water = False
        
        return underwater
    
    def _max_consecutive(self, trade_type):
        """Calculate max consecutive wins or losses"""
        max_count = 0
        current_count = 0
        
        for trade in self.trades:
            is_win = trade.get('pnl', 0) > 0
            
            if (trade_type == 'win' and is_win) or (trade_type == 'loss' and not is_win):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _calculate_monthly_consistency(self):
        """Calculate consistency across months"""
        # This would need timestamps on trades
        # Placeholder for now
        return 0.85  # 85% of months profitable
    
    def _calculate_kelly(self):
        """Calculate Kelly Criterion for optimal betting"""
        win_rate = self.metrics['consistency']['win_rate'] / 100
        avg_win = abs(self.metrics['consistency']['avg_win'])
        avg_loss = abs(self.metrics['consistency']['avg_loss'])
        
        if avg_loss == 0:
            return 0.25
        
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b
        
        return max(0, min(kelly, 0.25))  # Cap at 25%
    
    def _calculate_tail_ratio(self, returns):
        """Calculate Tail Ratio (right tail / left tail)"""
        if len(returns) == 0:
            return 1
        
        positive_returns = returns[returns > 0]
        negative_returns = abs(returns[returns < 0])
        
        right_tail = np.percentile(positive_returns, 95) if len(positive_returns) > 0 else 0
        left_tail = np.percentile(negative_returns, 95) if len(negative_returns) > 0 else 0.0001
        
        return right_tail / left_tail
    
    def _calculate_ulcer_index(self):
        """Calculate Ulcer Index (measure of downside risk)"""
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        squared_drawdowns = []
        peak = self.equity_curve[0]
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            squared_drawdowns.append(drawdown ** 2)
        
        ulcer = np.sqrt(np.mean(squared_drawdowns))
        return ulcer * 100
    
    def _calculate_sterling_ratio(self):
        """Calculate Sterling Ratio"""
        avg_dd = self.metrics['drawdowns']['avg_drawdown'] / 100
        cagr = self.metrics['returns']['cagr'] / 100
        
        return cagr / avg_dd if avg_dd > 0 else 0
    
    def _calculate_burke_ratio(self, returns):
        """Calculate Burke Ratio (penalizes large losses)"""
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        
        sum_squared = np.sum(negative_returns ** 2)
        burke = np.mean(returns) / np.sqrt(sum_squared / len(returns)) * np.sqrt(252)
        
        return burke
    
    def _print_analysis(self):
        """Print comprehensive analysis"""
        print("\n📊 RETURN METRICS:")
        print(f"  Total Return: {self.metrics['returns']['total_return']:.2f}%")
        print(f"  CAGR: {self.metrics['returns']['cagr']:.2f}%")
        print(f"  Sharpe Ratio: {self.metrics['returns']['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {self.metrics['returns']['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio: {self.metrics['returns']['calmar_ratio']:.2f}")
        
        print("\n📉 DRAWDOWN METRICS:")
        print(f"  Max Drawdown: {self.metrics['drawdowns']['max_drawdown']:.2f}%")
        print(f"  Avg Drawdown: {self.metrics['drawdowns']['avg_drawdown']:.2f}%")
        print(f"  Avg DD Duration: {self.metrics['drawdowns']['drawdown_duration']:.0f} days")
        print(f"  Underwater Periods: {self.metrics['drawdowns']['underwater_periods']}")
        
        print("\n🎯 CONSISTENCY METRICS:")
        print(f"  Win Rate: {self.metrics['consistency']['win_rate']:.1f}%")
        print(f"  Profit Factor: {self.metrics['consistency']['profit_factor']:.2f}")
        print(f"  Max Consecutive Wins: {self.metrics['consistency']['max_consecutive_wins']}")
        print(f"  Max Consecutive Losses: {self.metrics['consistency']['max_consecutive_losses']}")
        
        print("\n⭐ QUALITY METRICS:")
        print(f"  Kelly Criterion: {self.metrics['quality']['kelly_criterion']:.2%}")
        print(f"  Gain-to-Pain Ratio: {self.metrics['quality']['gain_to_pain_ratio']:.2f}")
        print(f"  Ulcer Index: {self.metrics['quality']['ulcer_index']:.2f}")
        print(f"  Sterling Ratio: {self.metrics['quality']['sterling_ratio']:.2f}")
        
        # Final rating
        rating = self._calculate_rating()
        print(f"\n🏆 SYSTEM RATING: {rating['grade']} ({rating['score']}/100)")
        print(f"   {rating['verdict']}")
    
    def _calculate_rating(self):
        """Calculate overall system rating"""
        score = 0
        
        # Return quality (30%)
        if self.metrics['returns']['sharpe_ratio'] > 2:
            score += 30
        elif self.metrics['returns']['sharpe_ratio'] > 1:
            score += 20
        elif self.metrics['returns']['sharpe_ratio'] > 0.5:
            score += 10
        
        # Drawdown quality (30%)
        if self.metrics['drawdowns']['max_drawdown'] < 5:
            score += 30
        elif self.metrics['drawdowns']['max_drawdown'] < 10:
            score += 20
        elif self.metrics['drawdowns']['max_drawdown'] < 15:
            score += 10
        
        # Consistency (20%)
        if self.metrics['consistency']['win_rate'] > 60:
            score += 20
        elif self.metrics['consistency']['win_rate'] > 55:
            score += 15
        elif self.metrics['consistency']['win_rate'] > 50:
            score += 10
        
        # Quality (20%)
        if self.metrics['quality']['gain_to_pain_ratio'] > 2:
            score += 20
        elif self.metrics['quality']['gain_to_pain_ratio'] > 1:
            score += 10
        
        # Determine grade
        if score >= 85:
            grade = "A+"
            verdict = "Exceptional - Professional Grade System"
        elif score >= 75:
            grade = "A"
            verdict = "Excellent - Highly Profitable System"
        elif score >= 65:
            grade = "B+"
            verdict = "Good - Solid Performance"
        elif score >= 55:
            grade = "B"
            verdict = "Adequate - Needs Refinement"
        elif score >= 45:
            grade = "C"
            verdict = "Marginal - Significant Improvements Needed"
        else:
            grade = "F"
            verdict = "Poor - System Not Viable"
        
        return {
            'score': score,
            'grade': grade,
            'verdict': verdict
        }