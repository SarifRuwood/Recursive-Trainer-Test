# Recursive Trainer Test

An advanced AI agent training framework for simulating cryptocurrency trading strategies in a realistic, controlled environment.

## 🎯 Project Goals

- **Research-driven**: Explore how different RL agent architectures perform in market simulation
- **Meta-learning**: Build agents that can train other agents to optimize trading strategies
- **Rigorous testing**: Compare multiple agent architectures against realistic market data
- **Scalability**: Design modular architecture for easy expansion to multiple assets/timeframes
- **Analysis**: Comprehensive visualization and metrics to understand what strategies actually work

## 📋 Project Structure

```
Recursive-Trainer-Test/
├── config.py                   # Configuration management
├── main.py                     # Main entry point
├── data/
│   ├── __init__.py
│   ├── data_fetcher.py        # Market data utilities
│   └── historical/            # Downloaded OHLCV data
├── environments/
│   ├── __init__.py
│   └── trading_env.py         # Custom Gym environment
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Base agent interface
│   ├── dqn_agent.py           # DQN implementation
│   └── ppo_agent.py           # PPO implementation
├── training/
│   ├── __init__.py
│   └── train.py               # Training runner
├── evaluation/
│   ├── __init__.py
│   └── backtest.py            # Backtesting framework
├── visualization/
│   ├── __init__.py
│   └── plots.py               # Results visualization
├── utils/
│   ├── __init__.py
│   ├── logger.py              # Logging utilities
│   └── data_utils.py          # File utilities
├── scripts/
│   ├── __init__.py
│   └── meta_trainer_demo.py   # Meta-learning demo
├── requirements.txt           # Project dependencies
├── .gitignore
└── README.md                  # This file
```

## 🛠️ Tech Stack

- **Framework**: Stable Baselines3 + Gymnasium (formerly OpenAI Gym)
- **RL Algorithms**: PPO, DQN, A2C, SAC
- **Data Source**: CCXT API + Historical OHLCV data
- **Language**: Python 3.10+
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Development**: Jupyter, IPython

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd Recursive-Trainer-Test
pip install -r requirements.txt
```

### 2. Run Main Demo

Train and compare PPO and DQN agents on synthetic data:

```bash
python main.py
```

This will:
- Generate synthetic market data
- Create a trading environment
- Train PPO and DQN agents
- Run backtests on test data
- Generate comparison visualizations
- Save results and logs

### 3. Run Meta-Trainer Demo

See recursive training in action:

```bash
python scripts/meta_trainer_demo.py
```

This demonstrates:
- Multi-generation agent evolution
- Hyperparameter exploration
- Cross-generation performance tracking

## 🎮 Core Components

### Trading Environment (`environments/trading_env.py`)

Custom Gymnasium environment simulating realistic crypto trading:

```python
from environments import CryptoTradingEnv
from data import generate_synthetic_data

data = generate_synthetic_data(n_candles=1000)
env = CryptoTradingEnv(
    data=data,
    initial_capital=10000.0,
    lookback_window=24,
    spread=0.0005,
    slippage=0.001,
)
```

**Features:**
- OHLCV data-driven price movements
- Configurable spreads and slippage
- Portfolio tracking (cash + holdings)
- Meaningful reward signals
- Actions: [hold=0, buy=1, sell=2]

### Agent Architectures

#### PPO Agent (`agents/ppo_agent.py`)

Proximal Policy Optimization agent:

```python
from agents import PPOAgent

agent = PPOAgent(
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)
agent.train(total_timesteps=100000)
```

#### DQN Agent (`agents/dqn_agent.py`)

Deep Q-Network agent:

```python
from agents import DQNAgent

agent = DQNAgent(
    env=env,
    learning_rate=1e-4,
    buffer_size=100000,
    exploration_final_eps=0.05,
)
agent.train(total_timesteps=100000)
```

### Meta-Trainer (`scripts/meta_trainer_demo.py`)

Recursive trainer that learns optimal agent configurations:

```python
from scripts.meta_trainer_demo import MetaTrainer

meta_trainer = MetaTrainer(config, data)
results = meta_trainer.evolve(n_generations=3)
```

## 📊 Data Sources

### Synthetic Data (Demo)

For quick testing and reproducibility:

```python
from data import generate_synthetic_data

data = generate_synthetic_data(
    n_candles=1000,
    start_price=50000.0,
    volatility=0.02,
    seed=42,
)
```

### Real Market Data

Fetch real data from exchanges:

```python
from data import load_or_fetch_data

data = load_or_fetch_data(
    symbol='BTC/USDT',
    start_date='2022-01-01',
    end_date='2024-06-21',
    timeframe='1h',
    use_cache=True,
)
```

## 📈 Evaluation & Visualization

### Backtesting

```python
from evaluation import BacktestRunner

backtester = BacktestRunner(test_env)
results = backtester.backtest_agent(agent, n_episodes=10)

print(f"Mean return: {results['mean_return']:.4f}")
print(f"Sharpe ratio: {results['sharpe_ratio']:.4f}")
print(f"Win rate: {results['win_rate']:.4f}")
```

### Metrics Tracked

- **Mean/Std Return**: Average return per episode
- **Win Rate**: % of episodes with positive returns
- **Sharpe Ratio**: Risk-adjusted returns
- **Max/Min Return**: Best and worst episode returns
- **Mean Final Portfolio Value**: Final account balance

### Visualization

```python
from visualization import ResultsVisualizer

visualizer = ResultsVisualizer(output_dir='results/plots')
visualizer.plot_agent_comparison(comparison_df)
visualizer.plot_metrics_heatmap(comparison_df)
visualizer.plot_portfolio_evolution(portfolio_values, prices)
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
from config import Config, DataConfig, EnvironmentConfig, AgentConfig

config = Config(
    data_config=DataConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2022-01-01",
    ),
    env_config=EnvironmentConfig(
        initial_capital=10000.0,
        spread=0.0005,
        slippage=0.001,
    ),
    agent_config=AgentConfig(
        agent_type="PPO",
        learning_rate=3e-4,
        policy_net_arch=[128, 128],
    ),
)
```

## 📝 Example Workflow

```python
# 1. Load data
from data import load_or_fetch_data
data = load_or_fetch_data('BTC/USDT', '2023-01-01', '2024-06-21')

# 2. Split data
train_data = data[:int(len(data)*0.7)]
test_data = data[int(len(data)*0.7):]

# 3. Create environments
from environments import CryptoTradingEnv
train_env = CryptoTradingEnv(train_data)
test_env = CryptoTradingEnv(test_data)

# 4. Train agents
from training import TrainingRunner
runner = TrainingRunner(config)
agent = runner.create_agent(train_env, agent_type="PPO")
runner.train_agent(agent, total_timesteps=1000000)

# 5. Backtest
from evaluation import BacktestRunner
backtester = BacktestRunner(test_env)
results = backtester.backtest_agent(agent, n_episodes=20)

# 6. Visualize
from visualization import ResultsVisualizer
visualizer = ResultsVisualizer()
visualizer.plot_agent_comparison(results_df)
```

## 📚 Key Features

✅ **Modular Design**: Easy to add new agent types, environments, or evaluation methods

✅ **Realistic Market Simulation**: Spreads, slippage, and realistic price action

✅ **Multiple RL Algorithms**: PPO, DQN implementations with more easy to add

✅ **Meta-Learning**: Agents that learn to train other agents

✅ **Comprehensive Metrics**: Sharpe ratio, win rate, returns distribution, etc.

✅ **Publication-Ready Visualizations**: Professional plots for analysis and reports

✅ **Production-Ready Logging**: Full audit trail of experiments

✅ **Reproducibility**: Seed control and caching for deterministic runs

## ⚠️ Important Disclaimers

### This is a Research/Learning Project

- **No real capital is at risk** - All trading is simulated
- **Results are on historical data** - Past performance ≠ future results
- **Goal is understanding RL**, not creating profitable systems
- **Markets are efficient** - Consistent profits are mathematically unlikely
- **Use for research only** - Never deploy to real trading without extensive validation

### Key Limitations

1. **Backtesting Bias**: Historical results may not predict future performance
2. **Slippage Assumptions**: Real market conditions are more complex
3. **No Transaction Costs**: Real trading includes fees, funding, etc.
4. **Overfitting Risk**: Agents may overfit to historical patterns
5. **Market Regime Changes**: Patterns change; past strategies may not work

## 🧪 Testing

Run tests (when available):

```bash
pytest tests/
```

## 📖 Documentation

- **API Reference**: See docstrings in each module
- **Examples**: Check `main.py` and `scripts/meta_trainer_demo.py`
- **Tutorials**: Look for `.ipynb` files in `notebooks/`

## 🤝 Contributing

This is a personal research project, but you're welcome to:

- Fork and adapt for your own research
- Report issues
- Suggest improvements

## 📄 License

MIT License - Feel free to use for research and education

## 🎓 Learning Resources

- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [CCXT Documentation](https://docs.ccxt.com/)
- [RL Theory (Sutton & Barto)](http://www.incompleteideas.net/book/the-book-2nd.html)

## 🚀 Next Steps

1. **Expand Agent Types**: Add SAC, A3C, other algorithms
2. **Multi-Asset Trading**: Train on multiple crypto pairs simultaneously
3. **Advanced Meta-Learning**: Learned hyperparameter optimization
4. **Real-Time Deployment**: Paper trading with live data
5. **Ensemble Methods**: Combine multiple agent predictions
6. **Risk Management**: Advanced portfolio optimization
7. **Interpretability**: Understand what agents learn

---

**Questions or ideas?** Open an issue to discuss!

**Remember**: This is research software. Use responsibly and never risk capital you can't afford to lose.
