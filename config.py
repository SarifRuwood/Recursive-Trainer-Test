"""
Configuration management for the Recursive Trainer Test project.

This module centralizes all configurable parameters for:
- Market data settings
- Trading environment parameters
- Agent hyperparameters
- Training configurations
"""

from dataclasses import dataclass
from typing import Dict, Any
import yaml
import os


@dataclass
class DataConfig:
    """Market data configuration."""
    # OHLCV data settings
    symbol: str = "BTC/USDT"  # Trading pair
    exchange: str = "binance"  # CCXT exchange
    timeframe: str = "1h"  # 1m, 5m, 1h, 1d, etc.
    start_date: str = "2022-01-01"  # Historical data start
    end_date: str = "2024-06-21"  # Historical data end
    
    # Data path
    data_dir: str = "data/historical"
    cache_enabled: bool = True


@dataclass
class EnvironmentConfig:
    """Trading environment configuration."""
    # Initial portfolio
    initial_capital: float = 10000.0  # Starting cash (USDT)
    initial_holdings: float = 0.0  # Starting crypto holdings
    
    # Market dynamics
    spread: float = 0.0005  # Bid-ask spread (0.05%)
    slippage: float = 0.001  # Slippage on execution (0.1%)
    
    # Reward shaping
    reward_scaling: float = 1.0
    penalty_bad_trade: float = -0.1
    
    # Episode settings
    lookback_window: int = 24  # Hours of historical data to observe
    max_steps_per_episode: int = 252  # Trading days per episode
    normalize_observations: bool = True


@dataclass
class AgentConfig:
    """RL Agent configuration."""
    # Common settings
    agent_type: str = "PPO"  # "DQN", "PPO", "A2C", "SAC"
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    
    # Network architecture
    policy_net_arch: list = None  # [64, 64] for 2 hidden layers
    value_net_arch: list = None
    
    # PPO specific
    n_steps: int = 2048  # Steps per epoch
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    
    # DQN specific
    buffer_size: int = 100000
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    
    def __post_init__(self):
        """Set defaults for network architectures."""
        if self.policy_net_arch is None:
            self.policy_net_arch = [128, 128]
        if self.value_net_arch is None:
            self.value_net_arch = [128, 128]


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    # Training duration
    total_timesteps: int = 1_000_000  # Total environment steps
    n_episodes: int = 100  # Episodes to train
    eval_episodes: int = 10  # Episodes for evaluation
    eval_freq: int = 10000  # Evaluate every N steps
    
    # Checkpointing
    save_freq: int = 50000
    model_dir: str = "models"
    log_dir: str = "logs"
    
    # Parallelization
    n_envs: int = 1  # Number of parallel environments
    seed: int = 42
    verbose: int = 1  # 0=quiet, 1=info, 2=debug


@dataclass
class MetaTrainerConfig:
    """Meta-learning trainer configuration."""
    # Meta-agent settings
    n_agent_configs: int = 5  # Number of agent configurations to try
    n_rounds: int = 3  # Rounds of meta-training
    
    # Hyperparameter search space
    learning_rates: list = None
    batch_sizes: list = None
    
    # Selection
    keep_top_k: int = 3  # Keep best K agents after each round
    
    def __post_init__(self):
        """Set defaults for search spaces."""
        if self.learning_rates is None:
            self.learning_rates = [1e-4, 3e-4, 1e-3]
        if self.batch_sizes is None:
            self.batch_sizes = [32, 64, 128]


class Config:
    """Master configuration class."""
    
    def __init__(
        self,
        data_config: DataConfig = None,
        env_config: EnvironmentConfig = None,
        agent_config: AgentConfig = None,
        training_config: TrainingConfig = None,
        meta_config: MetaTrainerConfig = None,
    ):
        """Initialize configuration with defaults or provided values."""
        self.data = data_config or DataConfig()
        self.env = env_config or EnvironmentConfig()
        self.agent = agent_config or AgentConfig()
        self.training = training_config or TrainingConfig()
        self.meta = meta_config or MetaTrainerConfig()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse each section
        data = DataConfig(**config_dict.get('data', {}))
        env = EnvironmentConfig(**config_dict.get('environment', {}))
        agent = AgentConfig(**config_dict.get('agent', {}))
        training = TrainingConfig(**config_dict.get('training', {}))
        meta = MetaTrainerConfig(**config_dict.get('meta_trainer', {}))
        
        return cls(data, env, agent, training, meta)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': self.data.__dict__,
            'environment': self.env.__dict__,
            'agent': self.agent.__dict__,
            'training': self.training.__dict__,
            'meta_trainer': self.meta.__dict__,
        }
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Default configuration instance
DEFAULT_CONFIG = Config()
