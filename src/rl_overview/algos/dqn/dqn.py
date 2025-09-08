from __future__ import annotations

"""
最小 DQN 训练骨架（Torch，可选依赖）。
特性：
- 经验回放、目标网络、ε-贪心探索
- 支持 Double DQN（可选）
- 针对离散状态的 one-hot 特征器（GridWorld 演示用）

若未安装 torch，则在调用时报提示。
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from ...replay.buffer import ReplayBuffer
from ...nn.torch_utils import is_torch_available, get_device


@dataclass
class DQNResult:
    policy: list[int]
    iters: int


def _one_hot(n: int, idx: np.ndarray | int) -> np.ndarray:
    arr = np.zeros((idx.shape[0] if hasattr(idx, "shape") else 1, n), dtype=np.float32)
    if hasattr(idx, "shape"):
        rows = np.arange(idx.shape[0])
        arr[rows, idx] = 1.0
    else:
        arr[0, int(idx)] = 1.0
    return arr


def train_dqn(
    env,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    replay_capacity: int = 10_000,
    start_learning_after: int = 1_000,
    target_sync_interval: int = 1_000,
    total_steps: int = 20_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 10_000,
    double_dqn: bool = False,
    dueling: bool = False,
    seed: int = 42,
) -> DQNResult:
    if not is_torch_available():  # pragma: no cover - 运行时检查
        raise RuntimeError("未安装 torch。请使用 `uv sync --extra torch` 安装后再运行 DQN。")

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from ...nn.dqn_net import build_tiny_qnet, build_dueling_qnet

    device = get_device()
    rng = np.random.default_rng(seed)

    # 这里假设离散状态，使用 one-hot 特征
    obs_dim = env.n_states
    n_actions = env.n_actions
    make_net = build_dueling_qnet if dueling else build_tiny_qnet
    online = make_net((obs_dim,), n_actions).to(device)
    target = make_net((obs_dim,), n_actions).to(device)
    target.load_state_dict(online.state_dict())
    optim_ = optim.Adam(online.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    rb = ReplayBuffer(replay_capacity)
    s = env.reset(seed=seed)
    steps = 0

    def epsilon_by_step(t):
        f = min(1.0, t / max(1, epsilon_decay_steps))
        return epsilon_start + (epsilon_end - epsilon_start) * f

    while steps < total_steps:
        eps = epsilon_by_step(steps)
        if rng.random() < eps:
            a = int(rng.integers(0, n_actions))
        else:
            with torch.no_grad():
                q = online(torch.from_numpy(_one_hot(obs_dim, np.array([s]))).to(device))
                a = int(torch.argmax(q, dim=-1).item())

        s2, r, done, _ = env.step(a)
        rb.push(s, a, r, s2, done)
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1))) if done else s2
        steps += 1

        if steps >= start_learning_after and len(rb) >= batch_size:
            S, A, R, S2, D = rb.sample(batch_size, rng)

            S_oh = torch.from_numpy(_one_hot(obs_dim, S.squeeze(-1))).to(device)
            S2_oh = torch.from_numpy(_one_hot(obs_dim, S2.squeeze(-1))).to(device)
            A_t = torch.from_numpy(A).long().to(device)
            R_t = torch.from_numpy(R).float().to(device)
            D_t = torch.from_numpy(D).float().to(device)

            q = online(S_oh).gather(1, A_t.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                if double_dqn:
                    next_actions = torch.argmax(online(S2_oh), dim=1)
                    next_q = target(S2_oh).gather(1, next_actions.view(-1, 1)).squeeze(1)
                else:
                    next_q = torch.max(target(S2_oh), dim=1).values
                target_q = R_t + (1.0 - D_t) * gamma * next_q

            loss = loss_fn(q, target_q)
            optim_.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
            optim_.step()

        if steps % target_sync_interval == 0:
            target.load_state_dict(online.state_dict())

    # 导出贪心策略（针对离散状态）
    policy = []
    with torch.no_grad():
        for s_idx in range(env.n_states):
            q = online(torch.from_numpy(_one_hot(obs_dim, np.array([s_idx]))).to(device))
            policy.append(int(torch.argmax(q, dim=-1).item()))

    return DQNResult(policy=policy, iters=steps)
