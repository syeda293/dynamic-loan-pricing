import numpy as np
import gymnasium as gym
from gymnasium import spaces

class LoanPricingEnv(gym.Env):
    RATES = np.linspace(0.05, 0.35, 13)  # 5% se 35%

    def __init__(self, pds, amounts, lgd=0.6, cost_funds=0.04, seed=0):
        super().__init__()
        self.pds, self.amounts = np.asarray(pds), np.asarray(amounts)
        self.lgd, self.cost = lgd, cost_funds
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(len(self.RATES))
        self.observation_space = spaces.Box(0, 1, shape=(2,), dtype=np.float32)
        self._i = 0

    def _obs(self):
        return np.array(
            [self.pds[self._i], min(self.amounts[self._i] / 20000, 1)],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._i = self.rng.integers(len(self.pds))
        return self._obs(), {}

    def step(self, action):
        rate = self.RATES[action]
        pd_, amt = self.pds[self._i], self.amounts[self._i]
        # rate zyada = borrower accept karne ki probability kam
        accept = self.rng.random() < 1 / (1 + np.exp(25 * (rate - 0.15)))
        reward = 0.0
        if accept:
            if self.rng.random() < pd_:
                reward = -self.lgd * amt          # default: loss
            else:
                reward = (rate - self.cost) * amt  # profit
        reward /= 1000
        self._i = self.rng.integers(len(self.pds))
        return self._obs(), reward, True, False, {}