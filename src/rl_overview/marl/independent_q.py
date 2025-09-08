from __future__ import annotations

"""
独立 Q 学习（Independent Q-learning）在矩阵博弈上的最小实现。
两位玩家各自更新自己的 Q 表。
"""

from typing import Tuple
import numpy as np
from rich.console import Console

console = Console()


def independent_q(
    game,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    episodes: int = 50_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    A1, A2 = game.n_actions_1, game.n_actions_2
    Q1 = np.zeros((A1, A2))  # Q1(a1, a2)
    Q2 = np.zeros((A1, A2))  # Q2(a1, a2) —— 对称定义，便于演示
    rng = np.random.default_rng(seed)

    for ep in range(1, episodes + 1):
        _ = game.reset(seed=int(rng.integers(0, 2**31 - 1)))
        # ε-贪心地从各自的“条件Q”中选择动作
        if rng.random() < epsilon:
            a1 = int(rng.integers(0, A1))
        else:
            # 这里将 Q1 的对手动作边际化为均匀，选择期望收益最大的 a1
            a1 = int(np.argmax(Q1.mean(axis=1)))
        if rng.random() < epsilon:
            a2 = int(rng.integers(0, A2))
        else:
            a2 = int(np.argmax(Q2.mean(axis=0)))

        _, (__, r2, done, _info2) = game.step(a1, a2)
        # 玩家1的收益与玩家2相反（零和默认），我们可以通过 game.R1 获取准确 r1
        r1 = float(game.R1[a1, a2])

        # 单步对局：直接向回报逼近
        Q1[a1, a2] += alpha * (r1 - Q1[a1, a2])
        Q2[a1, a2] += alpha * (r2 - Q2[a1, a2])

        if ep % max(1, episodes // 10) == 0:
            console.log(f"Independent Q 进度 {ep}/{episodes}")

    return Q1, Q2

