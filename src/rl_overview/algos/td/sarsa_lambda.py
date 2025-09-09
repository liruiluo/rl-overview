from __future__ import annotations

from typing import List, Tuple

import numpy as np
from ..common import epsilon_greedy_action, greedy_action
from rich.console import Console

console = Console()


def sarsa_lambda(
    env,
    gamma: float = 0.99,
    lam: float = 0.9,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    episodes: int = 2000,
    max_steps: int = 1000,
    seed: int = 42,
    logger=None,
) -> Tuple[List[int], np.ndarray, int]:
    """
    Sarsa(λ)：资格迹版本的 on-policy TD 控制。
    返回：贪心策略、Q(s,a)、回合数。
    """
    S, A = env.n_states, env.n_actions
    Q = np.zeros((S, A), dtype=np.float64)
    rng = np.random.default_rng(seed)

    for ep in range(1, episodes + 1):
        E = np.zeros((S, A), dtype=np.float64)
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        a = epsilon_greedy_action(Q, s, epsilon, A, rng)
        ep_ret = 0.0
        ep_len = 0
        for _ in range(max_steps):
            s2, r, done, _ = env.step(a)
            a2 = epsilon_greedy_action(Q, s2, epsilon, A, rng)
            td_err = r + (0.0 if done else gamma * Q[s2, a2]) - Q[s, a]
            E[s, a] += 1.0
            Q += alpha * td_err * E
            E *= gamma * lam
            s, a = s2, a2
            ep_ret += r
            ep_len += 1
            if done:
                break
        if logger is not None:
            logger.log(ep, ep_ret, ep_len)
        if ep % max(1, episodes // 10) == 0:
            console.log(f"Sarsa(lambda) 进度 {ep}/{episodes}")

    policy = [int(greedy_action(Q, s)) for s in range(S)]
    return policy, Q, episodes
