from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Trajectory:
    states: List[int]
    actions: List[int]
    rewards: List[float]


def rollouts(env, policy, episodes: int = 100, max_steps: int = 200, seed: int = 42) -> List[Trajectory]:
    rng = np.random.default_rng(seed)
    trajs: List[Trajectory] = []
    for _ in range(episodes):
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        states = [int(s)]
        actions = []
        rewards = []
        for _ in range(max_steps):
            a = int(policy(int(s)))
            s2, r, done, _ = env.step(a)
            actions.append(a)
            rewards.append(float(r))
            states.append(int(s2))
            s = s2
            if done:
                break
        trajs.append(Trajectory(states, actions, rewards))
    return trajs


def bag_of_states(traj: Trajectory, n_states: int) -> np.ndarray:
    x = np.zeros(n_states, dtype=np.float64)
    for s in traj.states:
        if 0 <= s < n_states:
            x[int(s)] += 1.0
    return x


def pref_pairs_from_trajs(trajs: List[Trajectory], noise_std: float = 0.0, seed: int = 0) -> List[Tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    pairs = []
    n = len(trajs)
    for i in range(0, n - 1, 2):
        t1, t2 = trajs[i], trajs[i + 1]
        r1 = float(sum(t1.rewards)) + rng.normal(0, noise_std)
        r2 = float(sum(t2.rewards)) + rng.normal(0, noise_std)
        y = 1 if r1 >= r2 else 0  # 1 表示 t1 优于 t2
        pairs.append((i, i + 1, y))
    return pairs


def train_logistic_pairwise(X: np.ndarray, pairs: List[Tuple[int, int, int]], lr: float = 1e-2, epochs: int = 500, l2: float = 1e-3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    w = rng.normal(0, 0.01, size=d)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    for _ in range(epochs):
        grad = np.zeros_like(w)
        for i, j, y in pairs:
            # 我们用 (x_i - x_j) 作为输入，目标 y ∈ {0,1}
            z = float(np.dot(w, X[i] - X[j]))
            p = sigmoid(z)
            # 负对数似然的梯度： (p - y) * (x_i - x_j)
            grad += (p - y) * (X[i] - X[j])
        grad = grad / max(1, len(pairs)) + l2 * w
        w -= lr * grad
    return w


def predict_pref_score(w: np.ndarray, x: np.ndarray) -> float:
    return float(np.dot(w, x))

