import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, prices):
        super(TradingEnv, self).__init__()

        self.prices = prices
        self.max_steps = len(prices) - 1
        self.current_step = 0

        # Observation: [current price, position (0/1)]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

        # Action: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Portfolio state
        self.cash = 1000.0
        self.position = 0  # 0 = no stock, 1 = holding
        self.stock_price = self.prices[0]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.current_step = 0
        self.cash = 1000.0
        self.position = 0
        self.stock_price = self.prices[0]
        obs = self._get_obs()
        return obs, {}
   

    def _get_obs(self):
        return np.array([self.stock_price, self.position], dtype=np.float32)

    def step(self, action):
        prev_price = self.stock_price
        self.current_step += 1

        if self.current_step >= self.max_steps:
            done = True
        else:
            done = False

        self.stock_price = self.prices[self.current_step]

        # Execute action
        reward = 0.0
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.cash -= self.stock_price
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            self.cash += self.stock_price
            reward = self.cash - 1000.0  # Profit relative to start

        return self._get_obs(), reward, done,False, {}

