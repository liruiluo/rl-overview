from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


ActionFn = Callable[[int], int]


def greedy_action(Q: np.ndarray, s: int) -> int:
    return int(np.argmax(Q[s]))


def epsilon_greedy_action(Q: np.ndarray, s: int, epsilon: float, n_actions: int, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    return greedy_action(Q, s)


@dataclass
class Episode:
    states: List[int]
    actions: List[int]
    rewards: List[float]  # r_t corresponds to transition s_t -> s_{t+1}


def run_episode(env, policy: ActionFn, max_steps: int, rng: np.random.Generator) -> Episode:
    s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    states: List[int] = [int(s)]
    actions: List[int] = []
    rewards: List[float] = []

    for _ in range(max_steps):
        a = policy(int(s))
        s2, r, done, _info = env.step(a)
        actions.append(int(a))
        rewards.append(float(r))
        states.append(int(s2))
        s = s2
        if done:
            break
    return Episode(states, actions, rewards)

