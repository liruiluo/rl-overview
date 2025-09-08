from __future__ import annotations

from typing import List, Tuple, Callable

import numpy as np
from ..common import greedy_action
from rich.console import Console

console = Console()


def td_lambda_prediction(
    env,
    policy: Callable[[int], int],
    gamma: float = 0.99,
    lam: float = 0.9,
    alpha: float = 0.1,
    episodes: int = 10_000,
    max_steps: int = 1_000,
    seed: int = 42,
) -> Tuple[List[float], int]:
    """
    TD(λ) 用于给定策略的状态价值预测（累积迹）。返回：V(s)、回合数。
    教学用最小实现（不含控制）。
    """
    S = env.n_states
    V = np.zeros(S, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(1, episodes + 1):
        E = np.zeros(S, dtype=np.float64)  # eligibility traces
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(max_steps):
            a = int(policy(int(s)))
            s2, r, done, _ = env.step(a)
            delta = r + (0.0 if done else gamma * V[s2]) - V[s]
            E[s] += 1.0
            V += alpha * delta * E
            E *= gamma * lam
            s = s2
            if done:
                break
        if ep % max(1, episodes // 10) == 0:
            console.log(f"TD(lambda) 进度 {ep}/{episodes}")
    return V.tolist(), episodes

