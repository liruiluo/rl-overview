from __future__ import annotations

import numpy as np


class BernoulliBandit:
    """
    K臂伯努利老虎机：每臂以各自的 p_k 给出1奖励，否则0。
    """

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float64)
        self.K = int(self.probs.shape[0])

    def pull(self, k: int) -> float:
        return float(np.random.random() < self.probs[k])

