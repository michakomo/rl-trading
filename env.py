import gym
import numpy as np
import pandas as pd

from typing import Tuple
from collections import deque
from sklearn.preprocessing import MinMaxScaler

class TradingEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, data: pd.DataFrame, window_size: int = 30, episode_size: int = 5000) -> None:

        self._raw_data = data
        self.data = MinMaxScaler().fit_transform(data) 

        self.window_size = window_size
        self.episode_size = episode_size

        # in position or out of position
        self.action_space = gym.spaces.Discrete(2) 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(window_size * (data.shape[1] + 1) + 1,), dtype=np.float64)

        self.init_balance = 100
        self.fee = 0.0004

        self._done = False

    
    def reset(self) -> np.ndarray:
        self.curr_idx = np.random.randint(self.window_size, self.data.shape[0] - self.episode_size)
        self.max_idx = self.curr_idx + self.episode_size

        self.action_history = deque(np.zeros(self.window_size) - 1, maxlen=self.window_size)
        self.balance = self.init_balance
        self._done = False

        self.in_position = False
        self.entry_price = None
        self.stop_price = None
        self.postion_size = None
        self.position_time = None
        # self.position_rr = -1.

        return self._get_obs()
    

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.data[self.curr_idx - self.window_size + 1: self.curr_idx + 1, :].flatten(),
            np.array(self.action_history),
            [self.balance]
        ])
    

    def _get_info(self) -> dict:
        return {"balance": self.balance,
                "position": self.in_position,
                "index": self._raw_data.index[self.curr_idx],
                "close": self._raw_data.iloc[self.curr_idx, 3],
                "entry_price": self.entry_price,
                "stop_price": self.stop_price,
                "position_size": self.postion_size
                }
    

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.action_history.append(action)
        reward = 0

        if self.position:
            # check if stop loss is hit
            if self._raw_data.iloc[self.curr_idx, 2] <= self.stop_price:
                # update balance
                self.balance += self.postion_size * (self.stop_price * (1 - self.fee) - self.entry_price * (1 + self.fee)) 
                reward = -2 if self.position_time < 10 else -1

                # reset position
                self.in_position = False
                self.entry_price = None
                self.stop_price = None
                self.postion_size = None
                self.position_time = None
            else:
                if action == 0: # close long position
                    # update balance
                    self.balance += self.postion_size * (self._raw_data.iloc[self.curr_idx, 3] * (1 - self.fee) - self.entry_price * (1 + self.fee))
                    
                    # reward as ratio of risk to reward of position
                    rr_ratio_base = np.abs(self.postion_size * (self.stop_price * (1 - self.fee) - self.entry_price * (1 + self.fee)))
                    reward = np.round((self.postion_size * (self._raw_data.iloc[self.curr_idx, 3] * (1 - self.fee) - self.entry_price * (1 + self.fee))), 3) / rr_ratio_base

                    if self.position_time < 10:
                        reward *= 0.5
 
                    # reset position
                    self.in_position = False
                    self.entry_price = None
                    self.stop_price = None
                    self.postion_size = None
                    self.position_time = None
                else: 
                    # hold long position
                    self.position_time += 1
        else:
            if action == 1:
                # open long position
                self.in_position = True
                self.entry_price = self._raw_data.iloc[self.curr_idx, 3] # enter at market on current close price
                self.stop_price = (1 - 0.0025) * self.entry_price # set stop price
                self.postion_size = min(np.round((self.balance * 5) / (0.25 * self.entry_price), 3), 5) # update position size l
                self.position_time = 0
            else:
                # do nothing
                pass

        if self.balance < 0.2 * self.init_balance: # stop trading if balance is less than 20% of initial balance
            self._done = True
            reward = -100 # large negative reward to discourage agent from going bankrupt

            return self._get_obs(), reward, self._done, self._get_info()
        
        if self.curr_idx == self.max_idx: # stop trading when episode ends
            self._done = True

            return self._get_obs(), reward, self._done, self._get_info()
        
        info = self._get_info()
        self.curr_idx += 1 # update current index

        return self._get_obs(), reward, self._done, info
    

    def render(self, mode: str = "human") -> None:
        return f"Balance: {self.balance:.2f}"


    def close(self) -> None:
        pass