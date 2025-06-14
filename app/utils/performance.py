import numpy as np

def compute_kpis(rewards):
    rewards = np.array(rewards)
    returns = np.cumsum(rewards)

    sharpe_ratio = np.mean(rewards) / (np.std(rewards) + 1e-9) * np.sqrt(252)
    cumulative_return = returns[-1]
    max_drawdown = np.max(np.maximum.accumulate(returns) - returns)

    return {
        "sharpe_ratio": round(float(sharpe_ratio), 4),
        "cumulative_return": round(float(cumulative_return), 2),
        "max_drawdown": round(float(max_drawdown), 2),
        "steps": len(rewards)
    }
