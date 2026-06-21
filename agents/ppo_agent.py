"""
PPO (Proximal Policy Optimization) agent implementation.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MlpPolicy
from .base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """
    PPO agent for trading using Stable Baselines3 PPO implementation.
    """
    
    def __init__(
        self,
        env: Any,
        name: str = "PPO_Agent",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        policy_net_arch: list = None,
        verbose: int = 0,
    ):
        """
        Initialize PPO agent.
        
        Args:
            env: Trading environment
            name: Agent name
            learning_rate: Learning rate
            n_steps: Steps per epoch
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            policy_net_arch: Network architecture [hidden_sizes]
            verbose: Verbosity level
        """
        config = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'policy_net_arch': policy_net_arch or [128, 128],
        }
        
        super().__init__(env, name, config)
        
        # Set network architecture
        policy_kwargs = {
            'net_arch': [dict(pi=config['policy_net_arch'], vf=config['policy_net_arch'])]
        }
        
        # Create PPO model
        self.model = PPO(
            MlpPolicy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log='logs/ppo',
        )
        self.verbose = verbose
    
    def train(self, total_timesteps: int) -> None:
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total environment steps to train
        """
        self.model.learn(total_timesteps=total_timesteps)
        self.total_timesteps += total_timesteps
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, Optional[np.ndarray]]:
        """
        Get action prediction.
        
        Args:
            observation: Current state
            deterministic: Use greedy or stochastic policy
        
        Returns:
            action, hidden_state
        """
        action, _state = self.model.predict(
            observation,
            deterministic=deterministic,
        )
        return action, _state
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get agent summary.
        
        Returns:
            Dictionary with agent information
        """
        summary = super().get_summary()
        summary['algorithm'] = 'PPO'
        summary['model_params'] = self.model.num_parameters()
        return summary
