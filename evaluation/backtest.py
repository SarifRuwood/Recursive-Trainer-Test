"""
Backtesting and evaluation utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path

from agents import BaseAgent
from environments import CryptoTradingEnv
from utils import get_logger


logger = get_logger(__name__)


class BacktestRunner:
    """
    Backtesting framework for comparing agent strategies.
    """
    
    def __init__(self, env: CryptoTradingEnv):
        """
        Initialize backtester.
        
        Args:
            env: Trading environment
        """
        self.env = env
        self.logger = get_logger(__name__)
        self.results = {}
    
    def backtest_agent(
        self,
        agent: BaseAgent,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Backtest agent strategy.
        
        Args:
            agent: Agent to backtest
            n_episodes: Number of episodes to run
            deterministic: Use deterministic policy
        
        Returns:
            Backtest results
        """
        self.logger.info(f"Backtesting {agent.name} over {n_episodes} episodes...")
        
        episode_results = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_data = {
                'episode': episode,
                'step': [],
                'price': [],
                'action': [],
                'portfolio_value': [],
                'cash': [],
                'holdings': [],
                'reward': [],
            }
            
            while not done:
                action, _ = agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_data['step'].append(info.get('current_step', 0))
                episode_data['price'].append(info.get('price', 0))
                episode_data['action'].append(action)
                episode_data['portfolio_value'].append(info.get('portfolio_value', 0))
                episode_data['cash'].append(info.get('cash', 0))
                episode_data['holdings'].append(info.get('holdings', 0))
                episode_data['reward'].append(reward)
            
            episode_results.append(episode_data)
        
        # Aggregate results
        all_results = self._aggregate_results(episode_results)
        self.results[agent.name] = all_results
        
        return all_results
    
    def _aggregate_results(self, episode_results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate episode results into summary metrics.
        
        Args:
            episode_results: List of episode data
        
        Returns:
            Aggregated metrics
        """
        final_values = []
        total_returns = []
        win_rates = []
        
        for episode_data in episode_results:
            final_value = episode_data['portfolio_value'][-1]
            initial_value = episode_data['portfolio_value'][0]
            
            final_values.append(final_value)
            total_returns.append((final_value - initial_value) / initial_value)
            win_rates.append(1.0 if final_value > initial_value else 0.0)
        
        metrics = {
            'mean_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values),
            'mean_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'win_rate': np.mean(win_rates),
            'max_return': np.max(total_returns),
            'min_return': np.min(total_returns),
            'sharpe_ratio': self._calculate_sharpe(total_returns),
        }
        
        return metrics
    
    @staticmethod
    def _calculate_sharpe(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate
        
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = np.array(returns) - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def compare_agents(self, agents: List[BaseAgent]) -> pd.DataFrame:
        """
        Compare multiple agents.
        
        Args:
            agents: List of agents to compare
        
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for agent in agents:
            if agent.name in self.results:
                row = {'agent': agent.name, **self.results[agent.name]}
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('mean_return', ascending=False)
    
    def export_results(self, output_path: str = 'results/backtest_results.csv') -> None:
        """
        Export backtest results to CSV.
        
        Args:
            output_path: Path to save results
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for agent_name, metrics in self.results.items():
            row = {'agent': agent_name, **metrics}
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Results exported to {output_path}")
