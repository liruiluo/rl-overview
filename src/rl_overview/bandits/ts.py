from __future__ import annotations

import numpy as np


def thompson_sampling_bernoulli(bandit, rounds: int = 1000, seed: int = 42):
    K = bandit.K
    rng = np.random.default_rng(seed)
    alpha = np.ones(K)
    beta = np.ones(K)
    rewards = []

    for t in range(rounds):
        theta = rng.beta(alpha, beta)
        k = int(np.argmax(theta))
        r = bandit.pull(k)
        rewards.append(r)
        if r > 0:
            alpha[k] += 1
        else:
            beta[k] += 1

    return {
        "alpha": alpha,
        "beta": beta,
        "avg_reward": float(np.mean(rewards)),
        "cum_reward": float(np.sum(rewards)),
    }

