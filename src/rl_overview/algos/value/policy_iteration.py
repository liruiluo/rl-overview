from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rich.console import Console

console = Console()


def policy_iteration(
    env,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_policy_eval_iters: int = 10_000,
    max_policy_iters: int = 1_000,
) -> Tuple[List[int], List[float], int]:
    """
    策略迭代（Policy Iteration）。
    - 使用显式世界模型 env.P, env.R。
    - 交替进行“策略评估（迭代法）”与“策略改进（贪心）”。
    返回：最优策略、价值函数、外层策略迭代轮数。
    """

    P, R = env.P, env.R
    S, A, _ = P.shape

    # 初始策略：对非终止状态随机（此处固定为向右），终止状态动作占位 0
    policy = np.zeros(S, dtype=np.int64)
    for s in range(S):
        if s not in env.terminal_states:
            policy[s] = 1  # 往右

    V = np.zeros(S, dtype=np.float64)

    for it in range(1, max_policy_iters + 1):
        # 策略评估：在当前 policy 下进行值迭代（迭代法求解 V^π）
        for k in range(max_policy_eval_iters):
            delta = 0.0
            for s in range(S):
                a = policy[s]
                v_new = np.sum(P[s, a] * (R[s, a] + gamma * V))
                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new
            if delta < theta:
                break

        # 策略改进：对每个状态采取对 Q(s,a) 的贪心改进
        policy_stable = True
        for s in range(S):
            old_a = policy[s]
            # 计算所有动作的 Q
            Q_s = np.zeros(A, dtype=np.float64)
            for a in range(A):
                Q_s[a] = np.sum(P[s, a] * (R[s, a] + gamma * V))
            policy[s] = int(np.argmax(Q_s))
            if policy[s] != old_a:
                policy_stable = False

        console.log(f"策略迭代 {it:4d} | 稳定: {policy_stable}")
        if policy_stable:
            break

    return policy.tolist(), V.tolist(), it

