# Dynamic Loan Pricing Model

Banks usually charge fixed interest rates, which overcharges safe borrowers and
underprices risky ones. This project prices each loan individually.

A LightGBM model estimates each borrower's probability of default (PD), explained
with SHAP. A PPO reinforcement learning agent then takes the PD and loan amount and
chooses an interest rate that maximizes expected profit, balancing interest income,
cost of funds, and loss given default. The RL policy is compared against a fixed-rate
baseline.

**Stack:** Python, LightGBM, SHAP, Gymnasium, Stable-Baselines3, Streamlit

**Note:** Borrower acceptance and default outcomes are simulated on the German Credit
dataset, so the results show the approach, not real bank performance.