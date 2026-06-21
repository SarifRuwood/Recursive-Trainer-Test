"""
Visualization utilities for training and backtest results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from pathlib import Path


class ResultsVisualizer:
    """
    Visualize training and backtest results.
    """
    
    def __init__(self, output_dir: str = 'results/plots'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_agent_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'mean_return',
        title: str = 'Agent Performance Comparison',
    ) -> None:
        """
        Plot agent comparison bar chart.
        
        Args:
            comparison_df: DataFrame with agent metrics
            metric: Metric to plot
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if x > 0 else 'red' for x in comparison_df[metric]]
        ax.bar(comparison_df['agent'], comparison_df[metric], color=colors, alpha=0.7)
        
        ax.set_xlabel('Agent', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = Path(self.output_dir) / f"{metric}_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        plt.close()
    
    def plot_metrics_heatmap(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = None,
        title: str = 'Agent Metrics Heatmap',
    ) -> None:
        """
        Plot metrics heatmap.
        
        Args:
            comparison_df: DataFrame with agent metrics
            metrics: Metrics to include
            title: Plot title
        """
        if metrics is None:
            # Use numeric columns
            metrics = comparison_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Normalize metrics for better visualization
        data = comparison_df[['agent'] + metrics].set_index('agent')
        data_normalized = (data - data.min()) / (data.max() - data.min())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data_normalized.T, annot=data.T, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Normalized Score'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path(self.output_dir) / "metrics_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        plt.close()
    
    def plot_training_progress(
        self,
        rewards: List[float],
        agent_name: str = 'Agent',
        title: str = 'Training Progress',
    ) -> None:
        """
        Plot training progress over time.
        
        Args:
            rewards: List of episode rewards
            agent_name: Name of agent
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = np.arange(len(rewards))
        ax.plot(episodes, rewards, alpha=0.5, label='Episode Reward')
        
        # Add moving average
        window = max(1, len(rewards) // 20)
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        ax.plot(episodes, moving_avg, linewidth=2, label=f'Moving Avg (window={window})')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(f"{title} - {agent_name}", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = Path(self.output_dir) / f"{agent_name}_training_progress.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        plt.close()
    
    def plot_portfolio_evolution(
        self,
        portfolio_values: List[float],
        prices: List[float],
        agent_name: str = 'Agent',
    ) -> None:
        """
        Plot portfolio value evolution during backtest.
        
        Args:
            portfolio_values: List of portfolio values over time
            prices: List of asset prices over time
            agent_name: Name of agent
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Portfolio value
        ax1.plot(portfolio_values, linewidth=2, color='blue')
        ax1.fill_between(range(len(portfolio_values)), portfolio_values, alpha=0.3, color='blue')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.set_title(f"{agent_name} - Portfolio Evolution", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Price action
        ax2.plot(prices, linewidth=1, color='gray', alpha=0.7)
        ax2.fill_between(range(len(prices)), prices, alpha=0.3, color='gray')
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Price ($)', fontsize=12)
        ax2.set_title('Asset Price', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = Path(self.output_dir) / f"{agent_name}_portfolio_evolution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        plt.close()
    
    def plot_returns_distribution(
        self,
        returns: List[float],
        agent_name: str = 'Agent',
    ) -> None:
        """
        Plot distribution of returns.
        
        Args:
            returns: List of returns
            agent_name: Name of agent
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.4f}')
        ax.axvline(np.median(returns), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(returns):.4f}')
        
        ax.set_xlabel('Return', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f"{agent_name} - Returns Distribution", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = Path(self.output_dir) / f"{agent_name}_returns_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        plt.close()
