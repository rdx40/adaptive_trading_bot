# app/rl/trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional

class TradingEnv(gym.Env):
    def __init__(self, df, window_size=10, legacy_mode=False):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.legacy_mode = legacy_mode
        self.current_step = window_size
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0
        self.position_value = 0.0
        self.portfolio_value = self.initial_balance
        
        # Action space: 0=hold, 1=buy10%, 2=sell10%, 3=buy50%, 4=sell50%
        self.action_space = spaces.Discrete(5)
        
        # Observation space handling for backward compatibility
        if legacy_mode:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(window_size,),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(window_size + 3,),
                dtype=np.float32
            )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.position_value = 0.0
        self.portfolio_value = self.initial_balance
        return self._get_observation(), {}

    def _get_observation(self):
        prices = self.df['Close'].iloc[
            self.current_step - self.window_size:self.current_step
        ].values.astype(np.float32)
        
        if self.legacy_mode:
            return prices
        else:
            # Normalize prices to percentage changes
            norm_prices = prices / prices[0] - 1.0
            
            # Add portfolio state information
            position_pct = self.position / (self.position + self.balance + 1e-8)
            cash_pct = self.balance / self.initial_balance
            portfolio_pct = self.portfolio_value / self.initial_balance
            
            return np.concatenate([
                norm_prices,
                [position_pct, cash_pct, portfolio_pct]
            ])

    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        prev_value = self._calculate_portfolio_value(current_price)
        
        # Execute trade action
        if action == 1:  # Buy 10%
            trade_amount = min(self.balance * 0.1, self.balance) / current_price
            self.position += trade_amount
            self.balance -= trade_amount * current_price
        elif action == 2:  # Sell 10%
            trade_amount = min(self.position * 0.1, self.position)
            self.position -= trade_amount
            self.balance += trade_amount * current_price
        elif action == 3:  # Buy 50%
            trade_amount = min(self.balance * 0.5, self.balance) / current_price
            self.position += trade_amount
            self.balance -= trade_amount * current_price
        elif action == 4:  # Sell 50%
            trade_amount = min(self.position * 0.5, self.position)
            self.position -= trade_amount
            self.balance += trade_amount * current_price
        
        # Calculate new portfolio value and reward
        current_value = self._calculate_portfolio_value(current_price)
        reward = current_value - prev_value
        
        # Move to next time step
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= len(self.df) - 1
        if terminated:
            # Liquidate final position
            self.balance += self.position * current_price
            self.position = 0
            current_value = self._calculate_portfolio_value(current_price)
            reward = current_value - prev_value
        
        return self._get_observation(), reward, terminated, False, {}
    
    def _calculate_portfolio_value(self, current_price):
        """Calculate total portfolio value (cash + position value)"""
        self.position_value = self.position * current_price
        self.portfolio_value = self.balance + self.position_value
        return self.portfolio_value