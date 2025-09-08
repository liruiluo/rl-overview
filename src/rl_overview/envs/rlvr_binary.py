from __future__ import annotations

"""
RLVR（二值奇偶校验）环境：
- 回合长度 n_bits；每步选择动作 {0,1}，更新累计奇偶 parity。
- 观测 = (pos, parity) 离散化为一个状态索引；
- 结束时若 parity == target_parity 给 1.0 奖励，否则 0.0。
"""

from typing import Tuple


class BinaryParityEnv:
    def __init__(self, n_bits: int = 8, target_parity: int = 0):
        assert n_bits > 0
        assert target_parity in (0, 1)
        self.n_bits = int(n_bits)
        self.target_parity = int(target_parity)
        self.n_actions = 2
        self.state = None  # (pos, parity)

    @property
    def n_states(self) -> int:
        return (self.n_bits + 1) * 2

    @property
    def terminal_states(self):  # unused
        return ()

    def _encode(self, pos: int, parity: int) -> int:
        return pos * 2 + parity

    def reset(self, seed: int | None = None) -> int:
        self.state = (0, 0)
        return self._encode(*self.state)

    def step(self, action: int):
        assert self.state is not None
        pos, parity = self.state
        parity ^= (action & 1)
        pos += 1
        done = pos >= self.n_bits
        if done:
            r = 1.0 if parity == self.target_parity else 0.0
        else:
            r = 0.0
        self.state = (pos, parity)
        s_idx = self._encode(*self.state)
        return s_idx, float(r), bool(done), {}

