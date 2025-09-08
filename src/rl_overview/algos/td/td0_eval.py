from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rich.console import Console

console = Console()


def td0_prediction(
    env,
    policy,
    gamma: float = 0.99,
    alpha: float = 0.1,
    episodes: int = 2000,
    max_steps: int = 1000,
    seed: int = 42,
) -> Tuple[List[float], int]:
    """
    TD(0) 预测：给定策略 policy，对状态价值 V 做增量式估计。
    """
    V = np.zeros(env.n_states, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(1, episodes + 1):
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(max_steps):
            a = int(policy(int(s)))
            s2, r, done, _ = env.step(a)
            target = r + (0.0 if done else gamma * V[s2])
            V[s] += alpha * (target - V[s])
            s = s2
            if done:
                break
        if ep % max(1, episodes // 10) == 0:
            console.log(f"TD(0) 进度 {ep}/{episodes}")
    return V.tolist(), episodes

