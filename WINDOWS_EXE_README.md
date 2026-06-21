# Recursive Trainer Test - Windows .EXE Version

## 🎯 No Python, No pip, No Scripts - Just Click & Run!

This version packages everything as standalone `.exe` files for Windows. No installation headaches!

## 📦 Quick Start (3 Steps)

### Step 1: One-Time Setup

**Double-click** `build_all.bat`

This will:
1. Download and install Python dependencies
2. Build the executable files
3. Create all necessary folders

⏱️ Takes ~5-10 minutes the first time

### Step 2: Run the Main Trainer

**Double-click** `run_trainer.bat`

Or directly run: `dist\RecursiveTrainer_Main.exe`

This will:
- Generate synthetic market data
- Train PPO and DQN agents
- Run backtests
- Generate comparison charts
- Save results to `results/`

### Step 3: Run the Meta-Trainer (Optional)

**Double-click** `run_meta_trainer.bat`

Or directly run: `dist\RecursiveTrainer_MetaTrainer.exe`

This demonstrates recursive training: agents learning to train other agents!

---

## 📁 Folder Structure After Setup

```
Recursive-Trainer-Test/
├── build_all.bat              ← Run this first (one time)
├── run_trainer.bat            ← Run this to train agents
├── run_meta_trainer.bat       ← Run this for meta-training demo
│
├── dist/
│   ├── RecursiveTrainer_Main.exe        ← Standalone executable
│   └── RecursiveTrainer_MetaTrainer.exe ← Meta-trainer executable
│
├── results/
│   ├── backtest_results.csv
│   └── plots/
│       ├── mean_return_comparison.png
│       ├── win_rate_comparison.png
│       └── metrics_heatmap.png
│
├── logs/
│   └── *.log                  ← Detailed execution logs
│
├── models/
│   ├── PPO_Agent.zip          ← Trained model
│   └── DQN_Agent.zip          ← Trained model
│
└── [source files]
```

---

## 🚀 Usage

### For Training Agents

```
Double-click: run_trainer.bat
   ↓
Window opens and shows training progress
   ↓
Automatically generates results and charts
   ↓
Results saved to: results/
```

### For Meta-Training

```
Double-click: run_meta_trainer.bat
   ↓
Window opens showing multi-generation training
   ↓
Compares different agent configurations
   ↓
Results saved to: results/
```

---

## 📊 What Gets Generated

### Results Files
- `results/backtest_results.csv` - Numerical metrics
- `results/plots/mean_return_comparison.png` - Performance chart
- `results/plots/win_rate_comparison.png` - Win rate chart
- `results/plots/metrics_heatmap.png` - Detailed metrics heatmap

### Logs
- `logs/__main__.log` - Main execution log
- `logs/train.py.log` - Training details
- `logs/backtest.py.log` - Evaluation details

### Models
- `models/PPO_Agent.zip` - Trained PPO model
- `models/DQN_Agent.zip` - Trained DQN model

---

## 🔧 Customization

To customize parameters:

1. **Open** `config.py` with a text editor
2. **Modify** any settings (learning rate, initial capital, etc.)
3. **Save** the file
4. **Rebuild** executables: `python build_executables.py`
5. **Run** the new executable

---

## ⚙️ System Requirements

- **Windows 10** or newer
- **4 GB RAM** minimum (8 GB recommended)
- **500 MB disk space** for dependencies
- **2 GB disk space** for models and results
- **Internet connection** (for initial setup only)

---

## 🐛 Troubleshooting

### "build_all.bat didn't work"

1. Make sure Python is installed: https://www.python.org/downloads/
2. **Important:** During Python installation, check "Add Python to PATH"
3. Close and reopen Command Prompt
4. Try running `build_all.bat` again

### "The executable won't start"

Check the logs:
- Open `logs/` folder
- Look for `.log` files
- Open them with Notepad to see what went wrong

### "Out of memory" error

The training is using too much RAM. Solutions:
1. Close other applications
2. Reduce parameters in `config.py`:
   - Lower `n_steps` (e.g., 1024 instead of 2048)
   - Lower `buffer_size` (e.g., 50000 instead of 100000)
3. Rebuild: `python build_executables.py`

---

## 📝 Manual Rebuild

If you modify `config.py` or the source code:

```batch
REM Option 1: Use batch file
build_all.bat

REM Option 2: Just rebuild executables
python build_executables.py
```

---

## 💡 Advanced Options

### Edit Configuration

Open `config.py` with any text editor and modify:

```python
# Example: Change initial capital
data_config=DataConfig(
    initial_capital=20000.0  # Changed from 10000.0
)
```

### Add Custom Agents

1. Create new agent in `agents/my_agent.py`
2. Modify `main.py` to use your agent
3. Rebuild: `python build_executables.py`

### Use Real Market Data

Edit `main.py` and uncomment:

```python
# Uncomment this to use real data:
data = load_or_fetch_data(
    symbol='BTC/USDT',
    start_date='2022-01-01',
    end_date='2024-06-21',
    timeframe='1h',
)
```

---

## 📚 Understanding the Results

### Metrics Explained

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Mean Return** | Average profit/loss per episode | > 0% |
| **Win Rate** | % of profitable episodes | > 50% |
| **Sharpe Ratio** | Risk-adjusted returns | > 1.0 |
| **Max Return** | Best episode | Positive |
| **Min Return** | Worst episode | Not too negative |

---

## ⚠️ Important Disclaimers

✅ **This is research software** - Perfect for learning RL and market dynamics

❌ **Do NOT use for real trading** - Simulated results don't predict real performance

⚠️ **Markets are efficient** - Consistent profits are mathematically unlikely

---

## 📞 Getting Help

1. Check `logs/` for error messages
2. Review `results/` to understand outputs
3. Read `config.py` comments for parameter explanations
4. Look at source code docstrings in the `.py` files

---

## 🎓 Learning Path

1. **Start:** Run `run_trainer.bat` to see it work
2. **Explore:** Check results in `results/` folder
3. **Customize:** Edit `config.py` and rebuild
4. **Experiment:** Try different parameters
5. **Advanced:** Look at `scripts/meta_trainer_demo.py` to understand meta-learning

---

## 🚀 Next Steps

- Run the trainer multiple times with different seeds (in config.py)
- Compare results in `results/backtest_results.csv`
- Analyze the generated charts
- Try modifying agent hyperparameters
- Experiment with different market data

---

**Ready?** Double-click `build_all.bat` and let's go! 🎯
