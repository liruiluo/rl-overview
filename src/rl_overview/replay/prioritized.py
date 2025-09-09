from __future__ import annotations

import numpy as np
from typing import List, Tuple


class PrioritizedReplayBuffer:
    """
    简化版“比例优先级”回放：
    - 线性数组 + 每次采样按累积和 O(N) 选取（教学用，易读）
    - priorities^alpha 作为权重；返回重要性采样权重 w ∝ (1/N * 1/p_i)^beta 归一化
    """

    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-3):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.s: List[object] = [None] * capacity  # 状态可为标量或ndarray
        self.a = np.empty(capacity, dtype=np.int64)
        self.r = np.empty(capacity, dtype=np.float64)
        self.s2: List[object] = [None] * capacity
        self.done = np.empty(capacity, dtype=np.bool_)
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.size = 0
        self.ptr = 0

    def __len__(self):
        return self.size

    def push(self, s, a: int, r: float, s2, done: bool, priority: float | None = None):
        idx = self.ptr
        self.s[idx] = s
        self.a[idx] = int(a)
        self.r[idx] = float(r)
        self.s2[idx] = s2
        self.done[idx] = bool(done)
        if priority is None:
            priority = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[idx] = float(priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()
        assert self.size >= batch_size
        prios = self.priorities[: self.size] + self.eps
        probs = prios ** self.alpha
        probs = probs / probs.sum()
        # O(N) 采样：多项分布
        idx = rng.choice(self.size, size=batch_size, p=probs)
        # IS 权重
        weights = (self.size * probs[idx]) ** (-beta)
        weights = weights / weights.max()

        def to_array(x):
            import numpy as np
            if isinstance(x, (int, np.integer)):
                return np.array([x], dtype=np.int64)
            return np.asarray(x)

        S = np.stack([to_array(self.s[i]) for i in idx])
        A = self.a[idx]
        R = self.r[idx]
        S2 = np.stack([to_array(self.s2[i]) for i in idx])
        D = self.done[idx].astype(np.float64)
        W = weights.astype(np.float32)
        return S, A, R, S2, D, W, idx

    def update_priorities(self, idx, prios):
        self.priorities[idx] = np.asarray(prios, dtype=np.float64)

