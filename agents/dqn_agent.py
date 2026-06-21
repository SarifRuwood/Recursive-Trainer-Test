"""
DQN (Deep Q-Network) agent implementation.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.policies import MlpPolicy
from .base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """
    DQN agent for trading using Stable Baselines3 DQN implementation.
    """
    
    def __init__(
        self,
        env: Any,
        name: str = "DQN_Agent",
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        exploration_fraction: float = 0.1,
        exploration_final_eps: float = 0.05,
        target_update_interval: int = 10000,
        learning_starts: int = 10000,
        batch_size: int = 32,
        gamma: float = 0.99,
        policy_net_arch: list = None,
        verbose: int = 0,
    ):
        """
        Initialize DQN agent.
        
        Args:
            env: Trading environment
            name: Agent name
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            exploration_fraction: Fraction of training for exploration
            exploration_final_eps: Final exploration epsilon
            target_update_interval: Update target network every N steps
            learning_starts: Steps before learning starts
            batch_size: Batch size for training
            gamma: Discount factor
            policy_net_arch: Network architecture [hidden_sizes]
            verbose: Verbosity level
        """
        config = {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'exploration_fraction': exploration_fraction,
            'exploration_final_eps': exploration_final_eps,
            'target_update_interval': target_update_interval,
            'learning_starts': learning_starts,
            'batch_size': batch_size,
            'gamma': gamma,
            'policy_net_arch': policy_net_arch or [128, 128],
        }
        
        super().__init__(env, name, config)
        
        # Set network architecture
        policy_kwargs = {'net_arch': config['policy_net_arch']}
        
        # Create DQN model
        self.model = DQN(
            MlpPolicy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            target_update_interval=target_update_interval,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log='logs/dqn',
        )
        self.verbose = verbose
    
    def train(self, total_timesteps: int) -> None:
        """
        Train the DQN agent.
        
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
        summary['algorithm'] = 'DQN'
        summary['model_params'] = self.model.num_parameters()
        return summary
