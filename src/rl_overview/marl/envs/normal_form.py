from __future__ import annotations

"""
最小“矩阵博弈”多智能体环境：两位玩家同时选动作，给定收益矩阵。
接口：
- reset(seed) -> 无观测(None)，回合开始
- step(a1, a2) -> (obs1, r1, done, info), (obs2, r2, done, info)
  这里 obs 均为 None（一次性对局），done=True。
"""

from typing import Tuple, Any, Dict
import numpy as np


class NormalFormGame:
    def __init__(self, payoff1: np.ndarray, payoff2: np.ndarray | None = None):
        payoff1 = np.asarray(payoff1, dtype=np.float64)
        if payoff2 is None:
            payoff2 = -payoff1  # 零和默认
        else:
            payoff2 = np.asarray(payoff2, dtype=np.float64)
            assert payoff2.shape == payoff1.shape
        self.R1 = payoff1
        self.R2 = payoff2
        self.n_actions_1 = payoff1.shape[0]
        self.n_actions_2 = payoff1.shape[1]

    def reset(self, seed: int | None = None):
        return None, None

    def step(self, a1: int, a2: int):
        r1 = float(self.R1[a1, a2])
        r2 = float(self.R2[a1, a2])
        done = True
        info: Dict[str, Any] = {}
        return (None, r1, done, info), (None, r2, done, info)

