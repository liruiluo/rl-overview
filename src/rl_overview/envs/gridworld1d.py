from __future__ import annotations

import numpy as np


class GridWorld1D:
    """
    一维网格世界（教学用最小环境）。

    - 状态 s ∈ {0, 1, ..., N-1}
    - 动作 a ∈ {0(左), 1(右)}
    - 转移：向左/右各移动一步，越界则停在边界
    - 终止：两端点 0 与 N-1 为终止状态
    - 奖励：除终止步外，每步奖励为 step_penalty；到达右端点获得 right_terminal_reward

    我们刻意避免依赖 Gym，让“世界模型”完全显式，便于与 Bellman 方程、值迭代等推导一一对应。
    """

    def __init__(self, n_states: int = 7, step_penalty: float = -0.01, right_terminal_reward: float = 1.0):
        assert n_states >= 2, "至少需要两个状态（两个端点）。"
        self.n_states = int(n_states)
        self.n_actions = 2  # 0: left, 1: right
        self.step_penalty = float(step_penalty)
        self.right_terminal_reward = float(right_terminal_reward)
        self.state = None  # 在线交互接口使用

        # 预构建转移与奖励张量，便于值迭代直接调用“已知世界模型”。
        # P[s, a, s_next] = 概率；R[s, a, s_next] = 奖励
        self.P, self.R = self._build_model()

    # -------------------------- 在线交互接口（Gym 式） --------------------------
    def reset(self, seed: int | None = None) -> int:
        """重置到中间状态，便于向任一方向探索。"""
        if seed is not None:
            rng = np.random.default_rng(seed)
            # 避开终止状态
            self.state = int(rng.integers(low=1, high=self.n_states - 1))
        else:
            self.state = (self.n_states - 1) // 2
        return self.state

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        assert self.state is not None, "请先调用 reset()。"
        s = self.state
        left_term, right_term = self.terminal_states
        if s in (left_term, right_term):
            # 终止后保持不变
            return s, 0.0, True, {}

        if action == 0:
            s_next = max(0, s - 1)
        else:
            s_next = min(self.n_states - 1, s + 1)

        if s_next == right_term:
            r = float(self.right_terminal_reward)
            done = True
        else:
            r = float(self.step_penalty)
            done = s_next in (left_term, right_term)

        self.state = s_next
        return s_next, r, done, {}

    @property
    def terminal_states(self) -> tuple[int, int]:
        return 0, self.n_states - 1

    def _build_model(self):
        N = self.n_states
        A = self.n_actions
        P = np.zeros((N, A, N), dtype=np.float64)
        R = np.zeros((N, A, N), dtype=np.float64)

        left_term, right_term = self.terminal_states

        for s in range(N):
            for a in range(A):
                # 终止状态：自环（episode 结束）。
                if s in (left_term, right_term):
                    P[s, a, s] = 1.0
                    R[s, a, s] = 0.0
                    continue

                # 非终止状态：确定性移动
                if a == 0:  # left
                    s_next = max(0, s - 1)
                else:  # right
                    s_next = min(N - 1, s + 1)

                P[s, a, s_next] = 1.0
                # 奖励：到达右端点给大额奖励；普通步给轻微负奖励（鼓励更快到达）。
                if s_next == right_term:
                    R[s, a, s_next] = self.right_terminal_reward
                else:
                    R[s, a, s_next] = self.step_penalty

        return P, R
