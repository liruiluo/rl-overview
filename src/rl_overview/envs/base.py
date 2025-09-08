from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple
import numpy as np


class ModelEnv(Protocol):
    """
    显式世界模型接口：提供 P、R 张量，用于动态规划/值迭代等。
    要求：
    - P: np.ndarray [S, A, S]
    - R: np.ndarray [S, A, S]
    - n_states: int
    - n_actions: int
    - terminal_states: tuple[int, ...]
    """

    P: np.ndarray
    R: np.ndarray
    n_states: int
    n_actions: int

    @property
    def terminal_states(self) -> Tuple[int, ...]:  # pragma: no cover - small utility
        ...


class StepEnv(Protocol):
    """
    在线交互接口（Gym 式）：
    - reset(seed) -> obs
    - step(action) -> (obs, reward, terminated, info)
    注意：为教学简洁，truncated 合并到 terminated（或放入 info）。
    """

    def reset(self, seed: int | None = None) -> Any:
        ...

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        ...

