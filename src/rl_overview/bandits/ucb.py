from __future__ import annotations

import numpy as np


def ucb1(bandit, rounds: int = 1000, c: float = 2.0, seed: int = 42):
    K = bandit.K
    rng = np.random.default_rng(seed)
    counts = np.zeros(K, dtype=np.int64)
    values = np.zeros(K, dtype=np.float64)
    rewards = []

    # 先各拉一次
    for k in range(K):
        r = bandit.pull(k)
        counts[k] += 1
        values[k] += (r - values[k])
        rewards.append(r)

    for t in range(K, rounds):
        ucb = values + np.sqrt(c * np.log(t + 1) / np.maximum(1, counts))
        k = int(np.argmax(ucb))
        r = bandit.pull(k)
        counts[k] += 1
        values[k] += (r - values[k]) / counts[k]
        rewards.append(r)

    return {
        "counts": counts,
        "values": values,
        "avg_reward": float(np.mean(rewards)),
        "cum_reward": float(np.sum(rewards)),
    }

