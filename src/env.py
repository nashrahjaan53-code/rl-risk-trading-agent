import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class RiskConstrainedTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, max_drawdown_limit: float = 0.15):
        super(RiskConstrainedTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_drawdown_limit = max_drawdown_limit 
    
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(4,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.peak_net_worth = self.initial_balance
        self.net_worth = self.initial_balance
        
        obs = self._get_observation()
        info = {}
        return obs, info

    def _get_observation(self):
        price = self.df.loc[self.current_step, 'Close']
        rsi = self.df.loc[self.current_step, 'RSI']
        macd = self.df.loc[self.current_step, 'MACD']
        current_drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth if self.peak_net_worth > 0 else 0.0
        return np.array([price, rsi, macd, current_drawdown], dtype=np.float32)

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        
   
        if action == 2:  
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
        elif action == 0: 
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth
            
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth
        step_return = (self.net_worth - self.initial_balance) / self.initial_balance
        reward = step_return
        terminated = False
        truncated = False
        
        if drawdown > self.max_drawdown_limit:
            reward -= 10.0  
            terminated = True  
            print(f"--> Risk Limit Breached! Drawdown: {drawdown:.2%}. Terminating episode.")
            
   
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            truncated = True  
            
        obs = self._get_observation()
        info = {'net_worth': self.net_worth, 'drawdown': drawdown}
        
        return obs, reward, terminated, truncated, info