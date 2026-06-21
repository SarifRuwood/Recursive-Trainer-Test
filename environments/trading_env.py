"""
Custom Gym-compatible cryptocurrency trading environment.

This environment simulates realistic crypto trading with:
- OHLCV price data
- Configurable spreads and slippage
- Portfolio tracking
- Meaningful reward signals
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces


class CryptoTradingEnv(gym.Env):
    """
    Cryptocurrency trading environment for RL agents.
    
    State: OHLCV data + portfolio information
    Actions: [hold=0, buy=1, sell=2]
    Reward: Portfolio value change + penalties
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        lookback_window: int = 24,
        spread: float = 0.0005,
        slippage: float = 0.001,
        reward_scaling: float = 1.0,
    ):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame with OHLCV data (Open, High, Low, Close, Volume)
            initial_capital: Starting capital in USDT
            lookback_window: Number of candles to use as state
            spread: Bid-ask spread as fraction
            slippage: Execution slippage as fraction
            reward_scaling: Scale factor for rewards
        """
        super().__init__()
        
        # Data
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        
        # Market dynamics
        self.spread = spread
        self.slippage = slippage
        self.reward_scaling = reward_scaling
        
        # State: normalized OHLCV (5 values) + portfolio info (3 values)
        # Lookback window flattened: lookback_window * 5 + 3
        state_size = lookback_window * 5 + 3
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_size,),
            dtype=np.float32
        )
        
        # Actions: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Episode state
        self.current_step = 0
        self.cash = initial_capital
        self.holdings = 0.0  # Amount of crypto held
        self.portfolio_value = initial_capital
        self.previous_portfolio_value = initial_capital
        self.trades = []  # Track trades for analysis
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
        
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Start from random point in data (at least lookback_window steps in)
        min_start = self.lookback_window + 1
        max_start = len(self.data) - 100  # Leave some data for episode
        
        if max_start <= min_start:
            self.current_step = min_start
        else:
            self.current_step = np.random.randint(min_start, max_start)
        
        # Reset portfolio
        self.cash = self.initial_capital
        self.holdings = 0.0
        self.portfolio_value = self.initial_capital
        self.previous_portfolio_value = self.initial_capital
        self.trades = []
        
        observation = self._get_observation()
        info = {'current_step': self.current_step}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of the environment.
        
        Args:
            action: 0=hold, 1=buy, 2=sell
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        current_price = self.data.loc[self.current_step, 'close']
        
        # Execute action
        if action == 1:  # Buy
            self._execute_buy(current_price)
        elif action == 2:  # Sell
            self._execute_sell(current_price)
        # action == 0: hold (do nothing)
        
        # Update portfolio value
        self.previous_portfolio_value = self.portfolio_value
        self.portfolio_value = self.cash + self.holdings * current_price
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Move to next step
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= len(self.data) - 1
        truncated = self.portfolio_value <= 0  # Bankruptcy
        
        observation = self._get_observation()
        info = {
            'current_step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'holdings': self.holdings,
            'price': current_price,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _execute_buy(self, price: float) -> None:
        """
        Execute a buy order with spread and slippage.
        
        Args:
            price: Current market price
        """
        # Execution price includes spread and slippage
        execution_price = price * (1 + self.spread + self.slippage)
        
        # Buy with all available cash
        if self.cash > 0:
            amount = self.cash / execution_price
            self.holdings += amount
            self.cash = 0
            
            self.trades.append({
                'step': self.current_step,
                'action': 'buy',
                'price': execution_price,
                'amount': amount,
            })
    
    def _execute_sell(self, price: float) -> None:
        """
        Execute a sell order with spread and slippage.
        
        Args:
            price: Current market price
        """
        # Execution price reduces due to spread and slippage
        execution_price = price * (1 - self.spread - self.slippage)
        
        # Sell all holdings
        if self.holdings > 0:
            proceeds = self.holdings * execution_price
            self.cash += proceeds
            self.holdings = 0
            
            self.trades.append({
                'step': self.current_step,
                'action': 'sell',
                'price': execution_price,
                'amount': proceeds,
            })
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward signal.
        
        Reward = portfolio gain/loss scaled and normalized
        """
        # Simple: reward portfolio value increase
        portfolio_change = self.portfolio_value - self.previous_portfolio_value
        reward = portfolio_change / self.initial_capital * self.reward_scaling
        
        return float(reward)
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.
        
        State = normalized OHLCV data (lookback_window) + portfolio info
        """
        # Get lookback window
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step + 1
        
        window_data = self.data.loc[start_idx:end_idx]
        
        # Normalize OHLCV by current close price
        current_close = self.data.loc[self.current_step, 'close']
        normalized_ohlcv = window_data[['open', 'high', 'low', 'close', 'volume']].values / current_close
        
        # Flatten OHLCV
        ohlcv_flat = normalized_ohlcv.flatten()
        
        # Pad if necessary
        target_size = self.lookback_window * 5
        if len(ohlcv_flat) < target_size:
            ohlcv_flat = np.pad(ohlcv_flat, (target_size - len(ohlcv_flat), 0))
        else:
            ohlcv_flat = ohlcv_flat[-target_size:]
        
        # Portfolio info: normalized cash, holdings, portfolio value
        portfolio_info = np.array([
            self.cash / self.initial_capital,
            self.holdings,
            self.portfolio_value / self.initial_capital,
        ])
        
        observation = np.concatenate([ohlcv_flat, portfolio_info])
        
        return observation.astype(np.float32)
    
    def render(self) -> None:
        """
        Render current state (simple console output).
        """
        current_price = self.data.loc[self.current_step, 'close']
        print(
            f"Step: {self.current_step} | "
            f"Price: ${current_price:.2f} | "
            f"Cash: ${self.cash:.2f} | "
            f"Holdings: {self.holdings:.4f} | "
            f"Portfolio: ${self.portfolio_value:.2f}"
        )
