from __future__ import annotations

from typing import List, Tuple

import numpy as np
from ..common import run_episode, Episode, epsilon_greedy_action, greedy_action
from rich.console import Console

console = Console()


def mc_control_every_visit(
    env,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    episodes: int = 10_000,
    max_steps: int = 1_000,
    seed: int = 42,
    logger=None,
) -> Tuple[List[int], np.ndarray, int]:
    """
    每次访问（every-visit）MC 控制（ε-贪心），返回：贪心策略、Q(s,a)、轨迹数。
    """
    S, A = env.n_states, env.n_actions
    Q = np.zeros((S, A), dtype=np.float64)
    returns_sum = np.zeros((S, A), dtype=np.float64)
    returns_count = np.zeros((S, A), dtype=np.int64)

    rng = np.random.default_rng(seed)

    for ep in range(1, episodes + 1):
        # ε-贪心策略（基于当前 Q）
        def policy(s: int) -> int:
            return epsilon_greedy_action(Q, s, epsilon, A, rng)

        traj: Episode = run_episode(env, policy, max_steps, rng)
        if logger is not None:
            G0 = 0.0
            g = 1.0
            for r in traj.rewards:
                G0 += g * r
                g *= gamma
            logger.log(ep, G0, len(traj.actions))

        # 计算每个时间步的回报 G_t，并对该回合中每次访问的 (s,a) 更新均值
        G = 0.0
        visited = set()  # every-visit：我们也可以不去重，直接对每次访问更新
        for t in reversed(range(len(traj.actions))):
            s_t = traj.states[t]
            a_t = traj.actions[t]
            r_tp1 = traj.rewards[t]
            G = gamma * G + r_tp1
            # every-visit：对每次访问都更新
            returns_sum[s_t, a_t] += G
            returns_count[s_t, a_t] += 1
            Q[s_t, a_t] = returns_sum[s_t, a_t] / max(1, returns_count[s_t, a_t])

        if ep % max(1, episodes // 10) == 0:
            console.log(f"MC 控制 进度 {ep}/{episodes}")

    policy_greedy = [int(greedy_action(Q, s)) for s in range(S)]
    return policy_greedy, Q, episodes
