from __future__ import annotations

from typing import List, Tuple

import numpy as np
from ..common import epsilon_greedy_action, greedy_action
from rich.console import Console

console = Console()


def q_learning(
    env,
    gamma: float = 0.99,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    episodes: int = 10_000,
    max_steps: int = 1_000,
    seed: int = 42,
    logger=None,
) -> Tuple[List[int], np.ndarray, int]:
    """
    Q-learning（off-policy TD 控制）。返回：贪心策略、Q(s,a)、回合数。
    """
    S, A = env.n_states, env.n_actions
    Q = np.zeros((S, A), dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(1, episodes + 1):
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_ret = 0.0
        ep_len = 0
        for _ in range(max_steps):
            a = epsilon_greedy_action(Q, s, epsilon, A, rng)
            s2, r, done, _ = env.step(a)
            best_next = 0.0 if done else np.max(Q[s2])
            target = r + gamma * best_next
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            ep_ret += r
            ep_len += 1
            if done:
                break
        if logger is not None:
            logger.log(ep, ep_ret, ep_len)
        if ep % max(1, episodes // 10) == 0:
            console.log(f"Q-learning 进度 {ep}/{episodes}")

    policy_greedy = [int(greedy_action(Q, s)) for s in range(S)]
    return policy_greedy, Q, episodes
