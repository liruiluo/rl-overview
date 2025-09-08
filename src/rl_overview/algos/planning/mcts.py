from __future__ import annotations

"""
最小 MCTS(UCT) 实现，面向 ModelEnv (提供 P,R)。
用于决策时规划：给定 s0，进行若干次模拟，返回首步动作。
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MCTSNode:
    s: int
    n_actions: int
    N: np.ndarray = field(init=False)
    Q: np.ndarray = field(init=False)
    children: dict = field(default_factory=dict)  # (a, s_next) -> MCTSNode

    def __post_init__(self):
        self.N = np.zeros(self.n_actions, dtype=np.int64)
        self.Q = np.zeros(self.n_actions, dtype=np.float64)


def mcts_action(env, s0: int, gamma: float = 0.99, num_simulations: int = 1000, max_depth: int = 20, c_ucb: float = 1.0) -> int:
    P, R = env.P, env.R
    S, A, _ = P.shape
    root = MCTSNode(s0, env.n_actions)
    rng = np.random.default_rng()

    def sample_next(s: int, a: int) -> int:
        probs = P[s, a]
        if probs.sum() <= 0:
            return s
        return int(rng.choice(np.arange(S), p=probs / probs.sum()))

    def rollout(s: int, depth: int) -> float:
        if depth >= max_depth or s in env.terminal_states:
            return 0.0
        a = int(rng.integers(0, A))
        s2 = sample_next(s, a)
        r = float(R[s, a, s2])
        return r + gamma * rollout(s2, depth + 1)

    def uct(node: MCTSNode, depth: int) -> float:
        if depth >= max_depth or node.s in env.terminal_states:
            return 0.0
        # 选择动作
        total_N = node.N.sum() + 1e-8
        ucb = node.Q + c_ucb * np.sqrt(np.log(total_N) / (node.N + 1e-8))
        a = int(np.argmax(ucb))
        s2 = sample_next(node.s, a)
        child_key = (a, s2)
        if child_key not in node.children:
            node.children[child_key] = MCTSNode(s2, env.n_actions)
            # 扩展后做一次 rollout
            reward = float(R[node.s, a, s2])
            G = reward + gamma * rollout(s2, depth + 1)
        else:
            reward = float(R[node.s, a, s2])
            G = reward + gamma * uct(node.children[child_key], depth + 1)

        # 回传
        node.N[a] += 1
        node.Q[a] += (G - node.Q[a]) / node.N[a]
        return G

    for _ in range(num_simulations):
        uct(root, 0)

    return int(np.argmax(root.Q))


def random_shooting_action(env, s0: int, horizon: int = 10, n_candidates: int = 100, gamma: float = 0.99, rng=None) -> int:
    if rng is None:
        rng = np.random.default_rng()
    P, R = env.P, env.R
    S, A, _ = P.shape

    def sample_next(s: int, a: int) -> int:
        probs = P[s, a]
        if probs.sum() <= 0:
            return s
        return int(rng.choice(np.arange(S), p=probs / probs.sum()))

    best_return = -1e9
    best_a0 = 0
    for _ in range(n_candidates):
        s = s0
        G = 0.0
        g = 1.0
        a0 = int(rng.integers(0, A))
        a_seq = [a0] + [int(rng.integers(0, A)) for _ in range(horizon - 1)]
        for a in a_seq:
            s2 = sample_next(s, a)
            r = float(R[s, a, s2])
            G += g * r
            g *= gamma
            s = s2
        if G > best_return:
            best_return = G
            best_a0 = a0
    return best_a0

