"""
Training loop for RL agents.
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from environments import CryptoTradingEnv
from agents import PPOAgent, DQNAgent
from utils import setup_logger, get_logger


logger = setup_logger(__name__)


class TrainingRunner:
    """
    Manages training of RL agents.
    """
    
    def __init__(self, config: Config):
        """
        Initialize training runner.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Create necessary directories
        Path(config.training.model_dir).mkdir(parents=True, exist_ok=True)
        Path(config.training.log_dir).mkdir(parents=True, exist_ok=True)
    
    def create_environment(self, data: pd.DataFrame) -> CryptoTradingEnv:
        """
        Create trading environment.
        
        Args:
            data: OHLCV data
        
        Returns:
            Trading environment
        """
        env = CryptoTradingEnv(
            data=data,
            initial_capital=self.config.env.initial_capital,
            lookback_window=self.config.env.lookback_window,
            spread=self.config.env.spread,
            slippage=self.config.env.slippage,
            reward_scaling=self.config.env.reward_scaling,
        )
        return env
    
    def create_agent(
        self,
        env: CryptoTradingEnv,
        agent_type: str = "PPO",
        agent_name: str = None,
    ) -> Any:
        """
        Create RL agent.
        
        Args:
            env: Trading environment
            agent_type: Type of agent (PPO, DQN)
            agent_name: Name for the agent
        
        Returns:
            Agent instance
        """
        if agent_name is None:
            agent_name = f"{agent_type}_Agent"
        
        if agent_type == "PPO":
            agent = PPOAgent(
                env=env,
                name=agent_name,
                learning_rate=self.config.agent.learning_rate,
                n_steps=self.config.agent.n_steps,
                batch_size=self.config.agent.batch_size,
                n_epochs=self.config.agent.n_epochs,
                gamma=self.config.agent.gamma,
                gae_lambda=self.config.agent.gae_lambda,
                clip_range=self.config.agent.clip_range,
                policy_net_arch=self.config.agent.policy_net_arch,
                verbose=self.config.training.verbose,
            )
        elif agent_type == "DQN":
            agent = DQNAgent(
                env=env,
                name=agent_name,
                learning_rate=self.config.agent.learning_rate,
                buffer_size=self.config.agent.buffer_size,
                exploration_fraction=self.config.agent.exploration_fraction,
                exploration_final_eps=self.config.agent.exploration_final_eps,
                batch_size=self.config.agent.batch_size,
                gamma=self.config.agent.gamma,
                policy_net_arch=self.config.agent.policy_net_arch,
                verbose=self.config.training.verbose,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return agent
    
    def train_agent(
        self,
        agent: Any,
        total_timesteps: int = None,
    ) -> None:
        """
        Train a single agent.
        
        Args:
            agent: Agent to train
            total_timesteps: Number of timesteps to train
        """
        if total_timesteps is None:
            total_timesteps = self.config.training.total_timesteps
        
        self.logger.info(
            f"Training {agent.name} for {total_timesteps} timesteps..."
        )
        
        agent.train(total_timesteps)
        
        self.logger.info(f"Training complete. Total timesteps: {agent.total_timesteps}")
    
    def evaluate_agent(
        self,
        agent: Any,
        eval_env: CryptoTradingEnv,
        n_episodes: int = None,
    ) -> Dict[str, float]:
        """
        Evaluate agent performance.
        
        Args:
            agent: Agent to evaluate
            eval_env: Environment for evaluation
            n_episodes: Number of episodes to evaluate
        
        Returns:
            Metrics dictionary
        """
        if n_episodes is None:
            n_episodes = self.config.training.eval_episodes
        
        self.logger.info(f"Evaluating {agent.name} over {n_episodes} episodes...")
        
        metrics = agent.evaluate(eval_env, n_episodes=n_episodes, deterministic=True)
        
        self.logger.info(
            f"Evaluation Results: "
            f"Mean Reward: {metrics['mean_reward']:.4f}, "
            f"Mean Final Value: ${metrics['mean_final_value']:.2f}"
        )
        
        return metrics
    
    def save_agent(self, agent: Any, agent_name: str = None) -> str:
        """
        Save agent model.
        
        Args:
            agent: Agent to save
            agent_name: Optional custom name
        
        Returns:
            Path to saved model
        """
        if agent_name is None:
            agent_name = agent.name
        
        model_path = os.path.join(self.config.training.model_dir, f"{agent_name}.zip")
        agent.save(model_path)
        
        self.logger.info(f"Agent saved to {model_path}")
        return model_path
