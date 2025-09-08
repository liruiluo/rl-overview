from __future__ import annotations

"""
Tiny Q-network skeleton for later DQN use.

保持文件不被默认导入，避免在未安装 torch 时触发 ImportError。
"""

from typing import Tuple


def build_tiny_qnet(obs_shape: Tuple[int, ...], n_actions: int):
    """
    延迟导入 torch，避免成为硬依赖。
    obs_shape: e.g., (obs_dim,)
    n_actions: 动作维度
    """
    import torch
    import torch.nn as nn

    obs_dim = int(obs_shape[0])

    class TinyQNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, n_actions),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    return TinyQNet()


def build_dueling_qnet(obs_shape: Tuple[int, ...], n_actions: int):
    import torch
    import torch.nn as nn

    obs_dim = int(obs_shape[0])

    class DuelingQNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(obs_dim, 128), nn.ReLU(),
            )
            self.V = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
            self.A = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_actions))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.body(x)
            v = self.V(h)
            a = self.A(h)
            q = v + a - a.mean(dim=1, keepdim=True)
            return q

    return DuelingQNet()
