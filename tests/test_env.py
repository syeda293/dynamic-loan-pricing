import numpy as np
from src.pricing_env import LoanPricingEnv

def test_env_step():
    env = LoanPricingEnv(np.array([0.1, 0.5]), np.array([5000, 10000]))
    obs, _ = env.reset()
    assert obs.shape == (2,)
    obs, reward, done, truncated, _ = env.step(0)
    assert done and isinstance(reward, float)