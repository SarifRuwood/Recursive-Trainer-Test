"""
Demo script for meta-agent training.

This demonstrates the recursive trainer concept where
an agent learns to train other agents optimally.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config, AgentConfig
from data import generate_synthetic_data
from environments import CryptoTradingEnv
from training import TrainingRunner
from evaluation import BacktestRunner
from utils import setup_logger


logger = setup_logger(__name__, log_dir='logs')


class MetaTrainer:
    """
    Meta-agent that trains and compares other agents.
    
    Demonstrates the recursive aspect: an agent that learns to
    configure and train other agents optimally.
    """
    
    def __init__(self, config: Config, data: pd.DataFrame):
        """
        Initialize meta-trainer.
        
        Args:
            config: Configuration
            data: Market data
        """
        self.config = config
        self.data = data
        self.logger = logger
        self.trained_agents = []
        self.results_history = []
    
    def generate_agent_configs(self, n_configs: int = 3) -> List[Dict]:
        """
        Generate diverse agent configurations to try.
        
        This simulates exploring the hyperparameter space.
        
        Args:
            n_configs: Number of configurations
        
        Returns:
            List of agent configurations
        """
        configs = []
        
        learning_rates = [1e-4, 3e-4, 1e-3]
        batch_sizes = [32, 64, 128]
        
        for i in range(min(n_configs, len(learning_rates))):
            config_dict = {
                'name': f"Config_{i}",
                'learning_rate': learning_rates[i],
                'batch_size': batch_sizes[i % len(batch_sizes)],
                'n_steps': 2048,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
            }
            configs.append(config_dict)
        
        return configs
    
    def train_generation(
        self,
        generation: int,
        agent_configs: List[Dict],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> Tuple[List, pd.DataFrame]:
        """
        Train a generation of agents with different configurations.
        
        Args:
            generation: Generation number
            agent_configs: List of agent configurations
            train_data: Training data
            test_data: Test data
        
        Returns:
            List of trained agents and results dataframe
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Generation {generation}: Training {len(agent_configs)} agents")
        self.logger.info(f"{'='*60}")
        
        runner = TrainingRunner(self.config)
        generation_agents = []
        generation_results = []
        
        for config_dict in agent_configs:
            agent_name = f"Gen{generation}_{config_dict['name']}"
            self.logger.info(f"\nTraining {agent_name}...")
            
            # Create environment
            train_env = CryptoTradingEnv(
                data=train_data,
                initial_capital=self.config.env.initial_capital,
                lookback_window=self.config.env.lookback_window,
            )
            
            # Create agent with config
            agent = runner.create_agent(
                train_env,
                agent_type="PPO",
                agent_name=agent_name,
            )
            
            # Update agent hyperparameters
            agent.config.update(config_dict)
            
            # Train (shorter for demo)
            runner.train_agent(agent, total_timesteps=20000)
            
            # Evaluate
            test_env = CryptoTradingEnv(
                data=test_data,
                initial_capital=self.config.env.initial_capital,
                lookback_window=self.config.env.lookback_window,
            )
            
            backtester = BacktestRunner(test_env)
            results = backtester.backtest_agent(agent, n_episodes=3)
            
            generation_agents.append(agent)
            generation_results.append({
                'agent': agent_name,
                'config': config_dict['name'],
                'learning_rate': config_dict['learning_rate'],
                **results
            })
            
            self.logger.info(f"Mean return: {results['mean_return']:.4f}")
        
        results_df = pd.DataFrame(generation_results)
        self.results_history.append(results_df)
        self.trained_agents.extend(generation_agents)
        
        return generation_agents, results_df
    
    def evolve(self, n_generations: int = 3) -> pd.DataFrame:
        """
        Run multi-generation evolution of agents.
        
        Args:
            n_generations: Number of generations to evolve
        
        Returns:
            Combined results from all generations
        """
        # Split data
        split_idx = int(len(self.data) * 0.7)
        train_data = self.data[:split_idx].reset_index(drop=True)
        test_data = self.data[split_idx:].reset_index(drop=True)
        
        for generation in range(n_generations):
            # Generate configurations (could be adaptive based on previous results)
            configs = self.generate_agent_configs(n_configs=3)
            
            # Train generation
            agents, results = self.train_generation(
                generation,
                configs,
                train_data,
                test_data,
            )
            
            # Log best performer
            best = results.loc[results['mean_return'].idxmax()]
            self.logger.info(f"Generation {generation} best: {best['agent']} (return: {best['mean_return']:.4f})")
        
        # Combine all results
        all_results = pd.concat(self.results_history, ignore_index=True)
        return all_results


def main():
    """
    Run meta-trainer demo.
    """
    logger.info("Starting Meta-Trainer Demonstration")
    logger.info("This shows recursive training: an agent learning to train other agents")
    
    # Setup
    config = Config()
    data = generate_synthetic_data(n_candles=1000, seed=42)
    
    logger.info(f"Data: {len(data)} candles")
    logger.info(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Run meta-training
    meta_trainer = MetaTrainer(config, data)
    all_results = meta_trainer.evolve(n_generations=2)  # Fewer for demo
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("META-TRAINING SUMMARY")
    logger.info("="*60)
    
    best_overall = all_results.loc[all_results['mean_return'].idxmax()]
    logger.info(f"\nBest agent overall: {best_overall['agent']}")
    logger.info(f"  Configuration: {best_overall['config']}")
    logger.info(f"  Mean return: {best_overall['mean_return']:.4f}")
    logger.info(f"  Win rate: {best_overall['win_rate']:.4f}")
    logger.info(f"  Sharpe ratio: {best_overall['sharpe_ratio']:.4f}")
    
    logger.info(f"\nTotal agents trained: {len(meta_trainer.trained_agents)}")
    logger.info("\n✓ Meta-trainer demonstration complete!")


if __name__ == "__main__":
    main()
