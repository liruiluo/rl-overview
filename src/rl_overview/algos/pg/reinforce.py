from __future__ import annotations

"""
REINFORCE（蒙特卡洛策略梯度）最小实现（Torch）。
针对离散状态（one-hot）与离散动作（softmax）。
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from ...nn.torch_utils import is_torch_available, get_device


@dataclass
class ReinforceResult:
    policy: list[int]
    iters: int


def train_reinforce(
    env,
    gamma: float = 0.99,
    lr: float = 1e-2,
    episodes: int = 2000,
    max_steps: int = 1000,
    seed: int = 42,
) -> ReinforceResult:
    if not is_torch_available():  # pragma: no cover
        raise RuntimeError("未安装 torch。请使用 `uv sync --extra torch` 后再运行 REINFORCE。")

    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = get_device()
    rng = np.random.default_rng(seed)

    # 观测自适应：离散(one-hot)或向量
    s0 = env.reset(seed=seed)
    if isinstance(s0, (int, np.integer)):
        discrete_obs = True
        obs_dim = int(getattr(env, "n_states"))
    else:
        discrete_obs = False
        if hasattr(env, "obs_shape") and env.obs_shape is not None:
            obs_dim = int(env.obs_shape[0])
        else:
            obs_dim = int(np.asarray(s0).shape[-1])
    n_actions = int(getattr(env, "n_actions"))

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 64), nn.ReLU(),
                nn.Linear(64, n_actions)
            )

        def forward(self, x):
            return self.net(x)

    pi = Policy().to(device)
    optim_ = optim.Adam(pi.parameters(), lr=lr)

    def obs_tensor(s):
        if discrete_obs:
            x = torch.zeros(obs_dim, dtype=torch.float32, device=device)
            x[int(s)] = 1.0
            return x
        else:
            return torch.tensor(np.asarray(s), dtype=torch.float32, device=device)

    steps = 0

    for ep in range(1, episodes + 1):
        logps = []
        rewards = []
        s = s0
        for t in range(max_steps):
            logits = pi(obs_tensor(s))
            dist = torch.distributions.Categorical(logits=logits)
            a = int(dist.sample().item())
            logps.append(dist.log_prob(torch.tensor(a, device=device)))
            s2, r, done, _ = env.step(a)
            rewards.append(float(r))
            s = s2
            steps += 1
            if done:
                break
        # 下一轮episode初始状态
        s0 = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

        # 计算回报序列 G_t 并做梯度上升
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        # 标准化回报有助于稳定
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        logps_t = torch.stack(logps)
        loss = -(logps_t * returns_t).sum()
        optim_.zero_grad(set_to_none=True)
        loss.backward()
        optim_.step()

    # 导出贪心策略
    policy = []
    with torch.no_grad():
        if discrete_obs:
            for s_idx in range(obs_dim):
                logits = pi(obs_tensor(s_idx))
                a = int(torch.argmax(logits).item())
                policy.append(a)

    return ReinforceResult(policy=policy, iters=steps)
