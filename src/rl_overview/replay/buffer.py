from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Transition:
    s: np.ndarray | int
    a: int
    r: float
    s2: np.ndarray | int
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.s: List[np.ndarray | int] = [None] * capacity  # type: ignore
        self.a = np.empty(capacity, dtype=np.int64)
        self.r = np.empty(capacity, dtype=np.float64)
        self.s2: List[np.ndarray | int] = [None] * capacity  # type: ignore
        self.done = np.empty(capacity, dtype=np.bool_)
        self.size = 0
        self.ptr = 0

    def push(self, s, a: int, r: float, s2, done: bool):
        self.s[self.ptr] = s
        self.a[self.ptr] = int(a)
        self.r[self.ptr] = float(r)
        self.s2[self.ptr] = s2
        self.done[self.ptr] = bool(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.size >= batch_size, "buffer 不足以采样"
        idx = rng.integers(0, self.size, size=batch_size)
        # 将可能的标量状态包装为数组；保留多维状态原样
        def to_array(x):
            if isinstance(x, (int, np.integer)):
                return np.array([x], dtype=np.int64)
            return np.asarray(x)
        S = np.stack([to_array(self.s[i]) for i in idx])
        A = self.a[idx]
        R = self.r[idx]
        S2 = np.stack([to_array(self.s2[i]) for i in idx])
        D = self.done[idx].astype(np.float64)
        return S, A, R, S2, D

