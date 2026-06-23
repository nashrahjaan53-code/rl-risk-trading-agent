import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class RiskConstrainedTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, max_drawdown_limit=0.15):
        super(RiskConstrainedTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_drawdown_limit = max_drawdown_limit  # 15% max allowable drawdown
        
        # Actions: 0 = Sell, 1 = Hold, 2 = Buy
        self.action_space = spaces.Discrete(3)
        
        # Observation: Price, RSI, and current portfolio drawdown
        # (Simplified to a 3-element vector for illustration)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.peak_net_worth = self.initial_balance
        self.net_worth = self.initial_balance
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Extract features for the current step
        price = self.df.loc[self.current_step, 'Close']
        rsi = self.df.loc[self.current_step, 'RSI'] if 'RSI' in self.df.columns else 50.0
        current_drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth if self.peak_net_worth > 0 else 0
        
        return np.array([price, rsi, current_drawdown], dtype=np.float32)

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        
        # Execute Action
        if action == 2: # Buy
            if self.balance > current_price:
                self.shares_held += 1
                self.balance -= current_price
        elif action == 0: # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
                
        # Update Net Worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # Track Drawdown
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth
        
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth
        
        # Calculate Reward & Apply Risk Penalty
        step_return = (self.net_worth - self.initial_balance) / self.initial_balance # Simple proxy
        reward = step_return
        
        terminated = False
        # If agent breaches risk limits, penalize heavily and terminate episode
        if drawdown > self.max_drawdown_limit:
            reward -= 5.0  # Harsh penalty for reckless risk management
            terminated = True
            
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            terminated = True
            
        truncated = False
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, {}