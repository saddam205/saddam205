"""
visualizer.py
Part of the app/backtesting module.
Advanced visualization for backtest results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class BacktestVisualizer:
    """
    Advanced visualizer for backtest results
    Creates professional charts for performance analysis
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn'):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_equity_curve(self, equity_curve: pd.DataFrame, 
                          benchmark: Optional[pd.Series] = None,
                          title: str = "Equity Curve") -> plt.Figure:
        """
        Plot equity curve with optional benchmark
        
        Args:
            equity_curve: DataFrame with 'equity' and 'timestamp' columns
            benchmark: Optional benchmark series
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot equity curve
        ax.plot(equity_curve['timestamp'], equity_curve['equity'], 
                label='Strategy', linewidth=2, color='blue')
        
        # Plot benchmark if provided
        if benchmark is not None:
            # Normalize benchmark to start at same level
            normalized_benchmark = benchmark / benchmark.iloc[0] * equity_curve['equity'].iloc[0]
            ax.plot(equity_curve['timestamp'], normalized_benchmark, 
                    label='Benchmark', linewidth=1.5, color='gray', alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown(self, equity_curve: pd.DataFrame) -> plt.Figure:
        """
        Plot drawdown chart
        
        Args:
            equity_curve: DataFrame with 'equity' and 'timestamp' columns
        """
        # Calculate drawdown
        cumulative = equity_curve['equity'].values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Fill drawdown area
        ax.fill_between(equity_curve['timestamp'], 0, drawdown, 
                        color='red', alpha=0.5, label='Drawdown')
        
        # Highlight worst drawdown
        worst_dd_idx = np.argmin(drawdown)
        worst_dd = drawdown[worst_dd_idx]
        ax.axhline(y=worst_dd, color='darkred', linestyle='--', 
                   alpha=0.7, label=f'Worst: {worst_dd:.1f}%')
        
        ax.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_returns_distribution(self, returns: pd.Series) -> plt.Figure:
        """
        Plot returns distribution with statistics
        
        Args:
            returns: Series of returns
        """
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1]))
        
        # Histogram with KDE
        axes[0].hist(returns, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        returns.plot.kde(ax=axes[0], color='red', linewidth=2)
        axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[0].set_title('Returns Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Return')
        axes[0].set_ylabel('Density')
        
        # Add statistics text
        stats_text = f"Mean: {returns.mean():.4f}\nStd: {returns.std():.4f}\nSkew: {returns.skew():.2f}\nKurt: {returns.kurtosis():.2f}"
        axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Q-Q plot for normality
        from scipy import stats as scipy_stats
        scipy_stats.probplot(returns, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_monthly_returns_heatmap(self, returns: pd.Series) -> plt.Figure:
        """
        Create monthly returns heatmap
        
        Args:
            returns: Series of returns with datetime index
        """
        # Convert to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        pivot = monthly_returns_df.pivot(index='year', columns='month', values='return')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color map with diverging colors
        cmap = sns.diverging_palette(10, 130, as_cmap=True)
        
        sns.heatmap(pivot, annot=True, fmt='.1%', cmap=cmap, center=0,
                   linewidths=0.5, ax=ax, cbar_kws={'label': 'Return (%)'})
        
        ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        plt.tight_layout()
        return fig
    
    def plot_trade_analysis(self, trades: List) -> plt.Figure:
        """
        Plot trade analysis charts
        
        Args:
            trades: List of trade objects
        """
        if not trades:
            return None
        
        # Extract trade data
        pnls = [t.pnl for t in trades]
        durations = [(t.exit_time - t.entry_time).days for t in trades]
        
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1]))
        
        # P&L distribution
        axes[0, 0].hist(pnls, bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('P&L Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('P&L ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Trade duration distribution
        axes[0, 1].hist(durations, bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Trade Duration Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Duration (days)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Cumulative P&L
        cumulative_pnl = np.cumsum(pnls)
        axes[1, 0].plot(cumulative_pnl, color='purple', linewidth=2)
        axes[1, 0].fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, alpha=0.3, color='purple')
        axes[1, 0].set_title('Cumulative P&L', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Cumulative P&L ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Win rate by month
        trade_dates = [t.exit_time for t in trades]
        monthly_wins = []
        monthly_months = []
        
        for i, date in enumerate(trade_dates):
            month_key = date.strftime('%Y-%m')
            if month_key not in monthly_months:
                monthly_months.append(month_key)
                monthly_wins.append(1 if pnls[i] > 0 else 0)
            else:
                idx = monthly_months.index(month_key)
                monthly_wins[idx] += 1 if pnls[i] > 0 else 0
        
        axes[1, 1].bar(range(len(monthly_months)), monthly_wins, color='orange', alpha=0.7)
        axes[1, 1].set_title('Winning Trades by Month', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Number of Winning Trades')
        axes[1, 1].set_xticks(range(len(monthly_months)))
        axes[1, 1].set_xticklabels(monthly_months, rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_dashboard(self, equity_curve: pd.DataFrame, 
                                   returns: pd.Series,
                                   trades: List) -> go.Figure:
        """
        Create interactive plotly dashboard
        
        Args:
            equity_curve: DataFrame with equity data
            returns: Series of returns
            trades: List of trade objects
        
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Equity Curve', 'Drawdown', 
                           'Returns Distribution', 'Monthly Returns',
                           'Cumulative Returns', 'Trade Analysis'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(x=equity_curve['timestamp'], y=equity_curve['equity'],
                      mode='lines', name='Equity', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Drawdown
        cumulative = equity_curve['equity'].values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(x=equity_curve['timestamp'], y=drawdown,
                      mode='lines', name='Drawdown', fill='tozeroy',
                      line=dict(color='red', width=1)),
            row=1, col=2
        )
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=50, name='Returns',
                        marker_color='blue', opacity=0.7),
            row=2, col=1
        )
        
        # Monthly returns heatmap
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        fig.add_trace(
            go.Heatmap(z=monthly_df.pivot(index='year', columns='month', values='return').values,
                      x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                      y=monthly_df['year'].unique(),
                      colorscale='RdYlGn', zmid=0, name='Monthly Returns'),
            row=2, col=2
        )
        
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(x=cumulative_returns.index, y=cumulative_returns,
                      mode='lines', name='Cumulative Returns',
                      line=dict(color='green', width=2)),
            row=3, col=1
        )
        
        # Trade scatter plot
        if trades:
            trade_pnls = [t.pnl for t in trades]
            trade_dates = [t.exit_time for t in trades]
            
            fig.add_trace(
                go.Scatter(x=trade_dates, y=trade_pnls,
                          mode='markers', name='Trade P&L',
                          marker=dict(size=8, color=trade_pnls, 
                                    colorscale='RdYlGn', showscale=True),
                          text=[f'P&L: ${p:.2f}' for p in trade_pnls]),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(height=900, showlegend=True, title_text="Backtest Dashboard")
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Return", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=3, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=3, col=2)
        
        return fig


class ChartGenerator:
    """Generate charts for reports"""
    
    @staticmethod
    def create_performance_summary(metrics: Dict) -> plt.Figure:
        """Create performance summary chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Metrics to display
        metric_names = ['Total Return', 'Sharpe Ratio', 'Sortino Ratio', 
                       'Win Rate', 'Profit Factor', 'Max Drawdown']
        metric_values = [
            metrics.get('total_return', 0),
            metrics.get('sharpe_ratio', 0),
            metrics.get('sortino_ratio', 0),
            metrics.get('win_rate', 0),
            metrics.get('profit_factor', 0),
            -metrics.get('max_drawdown', 0)  # Negative for better visualization
        ]
        
        # Create bar chart
        colors = ['green' if v >= 0 else 'red' for v in metric_values]
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}', ha='center', va='bottom')
        
        ax.set_title('Performance Summary', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_risk_radar(metrics: Dict) -> plt.Figure:
        """Create radar chart for risk metrics"""
        categories = ['Sharpe', 'Sortino', 'Calmar', 'Win Rate', 
                     'Profit Factor', 'Recovery']
        values = [
            metrics.get('sharpe_ratio', 0),
            metrics.get('sortino_ratio', 0),
            metrics.get('calmar_ratio', 0),
            metrics.get('win_rate', 0) / 100,
            min(metrics.get('profit_factor', 0) / 3, 1),
            metrics.get('recovery_factor', 0) / 5
        ]
        
        # Normalize values to 0-1 scale
        values = [min(max(v, 0), 1) for v in values]
        
        # Close the loop
        values += values[:1]
        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Risk-Return Radar Chart', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig