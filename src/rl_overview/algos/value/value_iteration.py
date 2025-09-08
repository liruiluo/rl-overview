from __future__ import annotations

from typing import Tuple, List

import numpy as np
from rich.console import Console

console = Console()


def value_iteration(
    env,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iters: int = 10_000,
) -> Tuple[List[int], List[float], int]:
    """
    值迭代（Value Iteration）。

    我们在“已知世界模型”的前提下，直接求解最优价值函数 V* 与最优策略 π*。
    公式对应 Bellman 最优性方程：

        V*(s) = max_a Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V*(s') ]

    参数：
    - env: 教学环境，需暴露 env.P, env.R 两个张量：
        P[s, a, s'] = 概率，R[s, a, s'] = 奖励
    - gamma: 折扣因子 γ ∈ [0,1)
    - theta: 收敛阈值（价值函数的最大变化量 < theta 视为收敛）
    - max_iters: 最大迭代步数（防止意外不收敛）

    返回：
    - policy: 最优策略（对每个状态给出贪心动作 argmax_a Q(s,a)）
    - values: 收敛后的状态价值 V(s)
    - iters: 实际迭代次数

    教学提示：
    - 终止状态的处理：让其在 P 上自环，奖励为 0；这样 Bellman 更新自然“停住”。
    - 与策略迭代的关系：值迭代可看作“每步都取贪心策略”的策略迭代极限形式。
    """

    P = env.P  # [S, A, S]
    R = env.R  # [S, A, S]
    S, A, _ = P.shape

    V = np.zeros(S, dtype=np.float64)
    Q = np.zeros((S, A), dtype=np.float64)

    for it in range(1, max_iters + 1):
        delta = 0.0

        # 逐状态更新 V(s)
        for s in range(S):
            # 计算动作价值 Q(s,a) = Σ_{s'} P * (R + γ V)
            for a in range(A):
                Q[s, a] = np.sum(P[s, a] * (R[s, a] + gamma * V))

            v_new = np.max(Q[s])
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new

        if it % 10 == 0 or delta < theta:
            console.log(f"迭代 {it:4d} | ΔV = {delta:.3e}")

        if delta < theta:
            break

    # 根据收敛后的 V 提取最优策略
    policy = [int(np.argmax(Q[s])) for s in range(S)]
    return policy, V.tolist(), it

