import gym
import pandas as pd
import numpy as np


class TradingEnv(gym.Env):
    """
    Custom OpenAI Gym environment for trading cryptocurrency perpetual futures.
    At the moment, only long positions and market orders are implemented.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, data_frame: pd.DataFrame):
        super(TradingEnv, self).__init__()

        self.data_frame = data_frame

        # [balance, position, rsi]
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(3,), dtype=np.float32)
        # [mkt buy, mkt sell] (at the end of a candle)
        self.action_space = gym.spaces.Discrete(2)

        self.done = False
        self.position = False
        self.init_balance = 100.

    
    def get_position_size(self, balance, risk, stop_pct, entry_price):
        return np.min(
            10, np.round((balance * risk) / (stop_pct * entry_price), 3)
        )


    def get_obs(self):
        return np.array([self.balance / self.init_balance, self.position.astype(float), self.data_frame["rsi"][self.curr_idx]])


    def get_info(self):    
        return {
            "balance": self.balance
        }


    def reset(self):
        self.curr_idx = np.random.randint(100, self.data_frame.shape[0] // 2)
        self.balance = self.init_balance
        self.position = False
        self.done = False

        return self.get_obs()

    
    def step(self, action):
        reward = 0
        curr_row = self.data_frame.iloc[self.curr_idx, :]

        if self.position:
            if curr_row["low"] < self.stop_price:
                self.position = False
                self.balance -= 0.0004 * self.stop_price
                self.balance += self.position_size * (self.stop_price - self.entry_price)
            ### Could add exiting position when we are >2R in profit here
            else:
                if action == 0:
                    pass
                else:
                    self.position = False
                    curr_price = curr_row["close"]
                    self.balance -= 0.0004 * curr_price
                    self.balance += self.position_size * (curr_price - self.entry_price)
        else:
            if action == 0:
                pass
            else:
                self.postion = True
                self.entry_price = curr_row["close"]
                self.stop_price = np.round(self.entry_price * (1 - 0.002), 1)
                self.position_size = self.get_position_size(balance=self.balance, risk=7, stop_pct=0.2, entry_price=self.entry_price)
                self.balance -= 0.0004 * self.entry_price
        
        if self.balance < 0.25 * self.init_balance:
            self.done = True
            reward = -1
    
        if self.balance > 2 * self.init_balance:
            self.done = True
            reward = 1

        info = self.get_info()
        self.curr_idx += 1

        return self.get_obs(), reward, self.done, info