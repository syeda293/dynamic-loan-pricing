import numpy as np
from stable_baselines3 import PPO
from src.pricing_env import LoanPricingEnv

def train(pds, amounts, steps=50_000):
    env = LoanPricingEnv(pds, amounts)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=steps)
    return model

def evaluate(policy_fn, pds, amounts, n=5000):
    env = LoanPricingEnv(pds, amounts, seed=123)
    obs, _ = env.reset()
    total = 0
    for _ in range(n):
        obs, r, *_ = env.step(policy_fn(obs))
        total += r
    return total / n

# Baseline: sab ko fixed 15% rate
fixed_policy = lambda obs: int(np.argmin(np.abs(LoanPricingEnv.RATES - 0.15)))