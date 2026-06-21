"""
Base agent class defining the interface for all RL agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
from stable_baselines3 import BasePolicy


class BaseAgent(ABC):
    """
    Abstract base class for trading agents.
    
    Defines interface that all agent implementations must follow.
    """
    
    def __init__(
        self,
        env: Any,
        name: str,
        config: Dict[str, Any],
    ):
        """
        Initialize base agent.
        
        Args:
            env: Gymnasium environment
            name: Agent name/identifier
            config: Hyperparameter configuration
        """
        self.env = env
        self.name = name
        self.config = config
        self.model = None
        self.total_timesteps = 0
        self.episode_rewards = []
    
    @abstractmethod
    def train(self, total_timesteps: int) -> None:
        """
        Train the agent.
        
        Args:
            total_timesteps: Number of environment steps to train
        """
        pass
    
    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, Optional[np.ndarray]]:
        """
        Get action prediction for an observation.
        
        Args:
            observation: Current environment state
            deterministic: If True, use best action; if False, sample
        
        Returns:
            action, hidden_state
        """
        pass
    
    def evaluate(
        self,
        env: Any,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate agent performance.
        
        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes to run
            deterministic: Use deterministic or stochastic policy
        
        Returns:
            Dictionary of metrics
        """
        episode_rewards = []
        episode_lengths = []
        final_values = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            final_values.append(info.get('portfolio_value', 0))
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_final_value': np.mean(final_values),
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards),
        }
    
    def save(self, path: str) -> None:
        """
        Save agent model.
        
        Args:
            path: Path to save model
        """
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path: str) -> None:
        """
        Load agent model.
        
        Args:
            path: Path to load model from
        """
        if self.model is not None:
            self.model = self.model.load(path, env=self.env)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of agent performance and configuration.
        
        Returns:
            Dictionary with agent info
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'total_timesteps': self.total_timesteps,
            'config': self.config,
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
        }
