# Trading System with Reinforcement Learning (PPO)

A complete system for algorithmic trading using Reinforcement Learning (PPO) integrated with QuantConnect LEAN and Interactive Brokers.

## Features

- Historical data download from IBKR
- Feature engineering with technical indicators
- Reinforcement Learning environment for trading
- PPO agent with multi-asset training
- Hyperparameter optimization with Optuna
- Integration with QuantConnect LEAN for backtesting and live trading

## Installation

```bash
# Clone the repository
git clone https://github.com/oarjones/trading-rl-ppo.git
cd trading-rl-ppo

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
