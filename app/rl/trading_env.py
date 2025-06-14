# app/rl/trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional
class TradingEnv(gym.Env):
    def __init__(self, df, window_size=10):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.current_step = self.window_size
        self.done = False
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0
        self.total_reward = 0
        self.action_space = gym.spaces.Discrete(3)  # hold, buy, sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size,),
            dtype=np.float32
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.current_step = self.window_size  # <-- this is key
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_profit = 0

        obs = self._get_observation()
        return obs, {}


    def _get_observation(self):
        obs = self.df['Close'].iloc[self.current_step - self.window_size:self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        reward = 0
        current_price = self.df['Close'].iloc[self.current_step]
    
        # Example logic for buy/sell/hold
        if action == 1:  # Buy
            if self.balance >= current_price:
                self.position += 1
                self.balance -= current_price
        elif action == 2:  # Sell
            if self.position > 0:
                self.position -= 1
                self.balance += current_price
                reward += current_price  # or profit
    
        self.current_step += 1
    
        # Done/terminated condition
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # Or handle early truncation if needed
    
        obs = self._get_observation()
        info = {}
    
        return obs, reward, terminated, truncated, info
    


