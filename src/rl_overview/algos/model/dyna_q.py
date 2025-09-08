from __future__ import annotations

"""
Dyna-Q（基于模型的规划+基于样本的学习）。
我们这里采用“学习模型”的版本：用计数近似 P，用均值近似 R。
对于 GridWorld1D，这个模型会快速收敛；也便于展示 Dyna 思想。
"""

from typing import List, Tuple
import numpy as np
from rich.console import Console

console = Console()


def dyna_q(
    env,
    gamma: float = 0.99,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    episodes: int = 1000,
    max_steps: int = 1000,
    planning_steps: int = 20,
    seed: int = 42,
) -> Tuple[List[int], np.ndarray, int]:
    S, A = env.n_states, env.n_actions
    Q = np.zeros((S, A), dtype=np.float64)
    rng = np.random.default_rng(seed)

    # 学习的模型：计数/均值
    counts = np.zeros((S, A, S), dtype=np.float64)
    rewards_sum = np.zeros((S, A, S), dtype=np.float64)

    for ep in range(1, episodes + 1):
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(max_steps):
            # 行动
            if rng.random() < epsilon:
                a = int(rng.integers(0, A))
            else:
                a = int(np.argmax(Q[s]))
            s2, r, done, _ = env.step(a)
            # Q-learning 更新
            best_next = 0.0 if done else np.max(Q[s2])
            Q[s, a] += alpha * (r + gamma * best_next - Q[s, a])
            # 更新模型
            counts[s, a, s2] += 1.0
            rewards_sum[s, a, s2] += r

            # 规划：从已见过的 (s,a) 中采样，模拟一步更新
            for _ in range(planning_steps):
                # 采样已出现的 (s,a)
                seen_idx = np.argwhere(counts.sum(axis=2) > 0)
                if seen_idx.size == 0:
                    break
                i = rng.integers(0, len(seen_idx))
                s_p, a_p = map(int, seen_idx[i])
                # 根据经验概率采样 s'
                probs = counts[s_p, a_p]
                probs = probs / probs.sum()
                s2_p = int(rng.choice(np.arange(S), p=probs))
                r_p = rewards_sum[s_p, a_p, s2_p] / max(1.0, counts[s_p, a_p, s2_p])
                best_next_p = 0.0 if (s2_p in env.terminal_states) else np.max(Q[s2_p])
                Q[s_p, a_p] += alpha * (r_p + gamma * best_next_p - Q[s_p, a_p])

            if done:
                break
            s = s2

        if ep % max(1, episodes // 10) == 0:
            console.log(f"Dyna-Q 进度 {ep}/{episodes}")

    policy = [int(np.argmax(Q[s])) for s in range(S)]
    return policy, Q, episodes

