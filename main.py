"""
Main entry point for the Recursive Trainer Test project.

This script demonstrates:
1. Fetching or generating market data
2. Creating a trading environment
3. Training multiple agent architectures
4. Comparing agent performance
5. Visualizing results
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from data import generate_synthetic_data, load_or_fetch_data
from environments import CryptoTradingEnv
from training import TrainingRunner
from evaluation import BacktestRunner
from visualization import ResultsVisualizer
from utils import setup_logger, ensure_dir


def main():
    """
    Main execution function.
    """
    # Setup logging
    logger = setup_logger(__name__, log_dir='logs')
    logger.info("Starting Recursive Trainer Test")
    
    # Load configuration
    config = Config()
    logger.info(f"Configuration loaded: {config.to_dict()}")
    
    # Create data directories
    ensure_dir(config.data.data_dir)
    ensure_dir(config.training.model_dir)
    ensure_dir(config.training.log_dir)
    
    # ==================== DATA PREPARATION ====================
    logger.info("\n" + "="*50)
    logger.info("STAGE 1: DATA PREPARATION")
    logger.info("="*50)
    
    # For demo, use synthetic data. In production, use real data:
    # data = load_or_fetch_data(
    #     symbol=config.data.symbol,
    #     start_date=config.data.start_date,
    #     end_date=config.data.end_date,
    #     timeframe=config.data.timeframe,
    # )
    
    logger.info("Generating synthetic market data for demonstration...")
    data = generate_synthetic_data(
        n_candles=1000,
        start_price=50000.0,
        volatility=0.02,
        seed=42,
    )
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Split data: 70% train, 30% test
    split_idx = int(len(data) * 0.7)
    train_data = data[:split_idx].reset_index(drop=True)
    test_data = data[split_idx:].reset_index(drop=True)
    
    logger.info(f"Train data: {len(train_data)} candles")
    logger.info(f"Test data: {len(test_data)} candles")
    
    # ==================== ENVIRONMENT SETUP ====================
    logger.info("\n" + "="*50)
    logger.info("STAGE 2: ENVIRONMENT SETUP")
    logger.info("="*50)
    
    train_env = CryptoTradingEnv(
        data=train_data,
        initial_capital=config.env.initial_capital,
        lookback_window=config.env.lookback_window,
        spread=config.env.spread,
        slippage=config.env.slippage,
    )
    
    test_env = CryptoTradingEnv(
        data=test_data,
        initial_capital=config.env.initial_capital,
        lookback_window=config.env.lookback_window,
        spread=config.env.spread,
        slippage=config.env.slippage,
    )
    
    logger.info(f"Training environment created")
    logger.info(f"  State space: {train_env.observation_space}")
    logger.info(f"  Action space: {train_env.action_space}")
    logger.info(f"  Initial capital: ${config.env.initial_capital}")
    
    # ==================== AGENT TRAINING ====================
    logger.info("\n" + "="*50)
    logger.info("STAGE 3: AGENT TRAINING")
    logger.info("="*50)
    
    runner = TrainingRunner(config)
    trained_agents = {}
    
    # Train PPO Agent
    logger.info("\nTraining PPO Agent...")
    ppo_env = CryptoTradingEnv(
        data=train_data,
        initial_capital=config.env.initial_capital,
        lookback_window=config.env.lookback_window,
    )
    ppo_agent = runner.create_agent(ppo_env, agent_type="PPO", agent_name="PPO_Agent")
    runner.train_agent(ppo_agent, total_timesteps=50000)  # Use smaller timesteps for demo
    trained_agents['PPO'] = ppo_agent
    runner.save_agent(ppo_agent)
    logger.info("PPO training complete!")
    
    # Train DQN Agent
    logger.info("\nTraining DQN Agent...")
    dqn_env = CryptoTradingEnv(
        data=train_data,
        initial_capital=config.env.initial_capital,
        lookback_window=config.env.lookback_window,
    )
    dqn_agent = runner.create_agent(dqn_env, agent_type="DQN", agent_name="DQN_Agent")
    runner.train_agent(dqn_agent, total_timesteps=50000)  # Use smaller timesteps for demo
    trained_agents['DQN'] = dqn_agent
    runner.save_agent(dqn_agent)
    logger.info("DQN training complete!")
    
    # ==================== BACKTESTING ====================
    logger.info("\n" + "="*50)
    logger.info("STAGE 4: BACKTESTING")
    logger.info("="*50)
    
    backtester = BacktestRunner(test_env)
    results_list = []
    
    for agent_name, agent in trained_agents.items():
        logger.info(f"\nBacktesting {agent_name}...")
        results = backtester.backtest_agent(agent, n_episodes=5, deterministic=True)
        results_list.append({'agent': agent_name, **results})
        logger.info(f"Results: {results}")
    
    results_df = pd.DataFrame(results_list)
    logger.info("\nBacktest Summary:")
    logger.info(results_df.to_string())
    
    # Export results
    backtester.export_results('results/backtest_results.csv')
    logger.info("Results exported to results/backtest_results.csv")
    
    # ==================== VISUALIZATION ====================
    logger.info("\n" + "="*50)
    logger.info("STAGE 5: VISUALIZATION")
    logger.info("="*50)
    
    visualizer = ResultsVisualizer(output_dir='results/plots')
    
    logger.info("Creating visualizations...")
    visualizer.plot_agent_comparison(
        results_df,
        metric='mean_return',
        title='Agent Performance - Mean Return',
    )
    visualizer.plot_agent_comparison(
        results_df,
        metric='win_rate',
        title='Agent Performance - Win Rate',
    )
    visualizer.plot_metrics_heatmap(results_df)
    
    logger.info("Visualizations created in results/plots/")
    
    # ==================== SUMMARY ====================
    logger.info("\n" + "="*50)
    logger.info("EXECUTION SUMMARY")
    logger.info("="*50)
    
    best_agent = results_df.loc[results_df['mean_return'].idxmax()]
    logger.info(f"Best performing agent: {best_agent['agent']}")
    logger.info(f"  Mean return: {best_agent['mean_return']:.4f}")
    logger.info(f"  Win rate: {best_agent['win_rate']:.4f}")
    logger.info(f"  Sharpe ratio: {best_agent['sharpe_ratio']:.4f}")
    
    logger.info("\n✓ Recursive Trainer Test completed successfully!")
    logger.info(f"Results saved to: {os.path.abspath('results')}")
    logger.info(f"Logs saved to: {os.path.abspath('logs')}")


if __name__ == "__main__":
    main()
