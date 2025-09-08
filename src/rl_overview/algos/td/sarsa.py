from __future__ import annotations

from typing import List, Tuple

import numpy as np
from ..common import run_episode, epsilon_greedy_action, greedy_action
from rich.console import Console

console = Console()


def sarsa(
    env,
    gamma: float = 0.99,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    episodes: int = 10_000,
    max_steps: int = 1_000,
    seed: int = 42,
) -> Tuple[List[int], np.ndarray, int]:
    """
    SARSA (on-policy TD 控制)。返回：贪心策略、Q(s,a)、回合数。
    """
    S, A = env.n_states, env.n_actions
    Q = np.zeros((S, A), dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(1, episodes + 1):
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        a = epsilon_greedy_action(Q, s, epsilon, A, rng)
        for _ in range(max_steps):
            s2, r, done, _ = env.step(a)
            a2 = epsilon_greedy_action(Q, s2, epsilon, A, rng)
            target = r + (0.0 if done else gamma * Q[s2, a2])
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
            if done:
                break
        if ep % max(1, episodes // 10) == 0:
            console.log(f"SARSA 进度 {ep}/{episodes}")

    policy_greedy = [int(greedy_action(Q, s)) for s in range(S)]
    return policy_greedy, Q, episodes

