# Risk-Constrained Reinforcement Learning Trading Agent

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Environment](https://img.shields.io/badge/Gymnasium-v1.0+-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline featuring a custom financial trading environment that integrates **strict risk management boundaries** into an object-oriented reinforcement learning frame. The agent uses Proximal Policy Optimization (PPO) to maximize financial returns while dynamically penalizing and terminating trading runs that violate realistic risk thresholds (e.g., maximum allowable drawdown).

##  Core Features
* **Custom Financial Gymnasium Environment:** Built from scratch to track portfolio states, update real-time net worth, asset counts, and cash allocations.
* **Feature Engineering Pipeline:** Built-in automated historical data retrieval using `yfinance` accompanied by indicators computed via `ta` (RSI, MACD, Bollinger Bands).
* **Risk Engine Integration:** Dynamically calculates maximum drawdown at every step, enforcing a hard penalty rule system.
* **Out-of-Sample Backtesting:** Splits datasets cleanly to ensure evaluation occurs entirely on unseen historical periods.

---

## Repository Architecture

```text
rl-risk-trading-agent/
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py   # Handles data downloads and technical indicators
│   └── env.py             # Custom risk-constrained trading environment
│
├── main.py                # Setup, training loop, and evaluation script
├── .gitignore             # Ignores pycache, models, and tensorboard logs
└── README.md              # Project documentation

```

---

## Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/rl-risk-trading-agent.git](https://github.com/YOUR_GITHUB_USERNAME/rl-risk-trading-agent.git)
cd rl-risk-trading-agent

```


2. **Install dependencies:**
```bash
pip install yfinance pandas numpy ta gymnasium stable-baselines3 torch tensorboard

```
##  Usage

To run the complete data acquisition, training workflow, and backtesting cycle, simply execute:

```bash
python main.py

```

### Tracking Training Progress (TensorBoard)

To view real-time reward curves, entropy losses, and policy gradients in your browser, run:

```bash
tensorboard --logdir=./tensorboard_logs/

```

---

##  How the Agent Learns Risk

The reward loop evaluates step returns but dynamically applies safety boundaries:

$$Reward = Return_t - \lambda \cdot Penalty_{risk}$$

If the portfolio drawdown slips past the initialized `max_drawdown_limit` (e.g., `0.10`), the environment immediately triggers a harsh negative penalty weight, stops further simulation, liquidates positions, and signals a `terminated` episode flag.

Over training timesteps, metrics show an increase in mean episode execution length (`ep_len_mean`) as the agent progressively discovers policy rules to adaptively maneuver market variance without tripping safety limits.

---

##  License

This project is open-source software licensed under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```powershell
git add README.md
git commit -m "Docs: Add production-ready README with status badges and metrics explanation"
git push
