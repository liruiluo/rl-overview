from __future__ import annotations

"""
最小 C51（Distributional DQN）实现，针对离散状态（one-hot）教学环境。
仅用于 GridWorld1D 等离散小环境，便于展示分布式价值思想。
"""

from dataclasses import dataclass
import numpy as np

from ...nn.torch_utils import is_torch_available, get_device


@dataclass
class C51Result:
    policy: list[int]
    iters: int


def build_c51_net(obs_dim: int, n_actions: int, n_atoms: int):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, n_actions * n_atoms)
            )
            self.n_actions = n_actions
            self.n_atoms = n_atoms

        def forward(self, x):
            logits = self.net(x)
            B = x.shape[0]
            return logits.view(B, self.n_actions, self.n_atoms)

    return Net()


def train_c51(env, gamma: float = 0.99, lr: float = 1e-3, batch_size: int = 64, replay_capacity: int = 10000, total_steps: int = 20000, start_learning_after: int = 1000, target_sync_interval: int = 1000, n_atoms: int = 51, v_min: float = -1.0, v_max: float = 1.0, seed: int = 42) -> C51Result:
    if not is_torch_available():
        raise RuntimeError("未安装 torch。`uv sync --extra torch` 后再运行 C51。")
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from ...replay.buffer import ReplayBuffer

    device = get_device()
    rng = np.random.default_rng(seed)

    # 离散观测
    obs_dim = int(getattr(env, "n_states"))
    n_actions = int(getattr(env, "n_actions"))

    def one_hot(idx):
        x = torch.zeros(obs_dim, dtype=torch.float32, device=device)
        x[int(idx)] = 1.0
        return x

    online = build_c51_net(obs_dim, n_actions, n_atoms).to(device)
    target = build_c51_net(obs_dim, n_actions, n_atoms).to(device)
    target.load_state_dict(online.state_dict())
    opt = optim.Adam(online.parameters(), lr=lr)

    rb = ReplayBuffer(replay_capacity)
    s = env.reset(seed=seed)
    steps = 0

    z = torch.linspace(v_min, v_max, n_atoms, device=device)
    delta_z = (v_max - v_min) / (n_atoms - 1)
    log_softmax = nn.LogSoftmax(dim=-1)

    def dist(q_logits):
        return torch.softmax(q_logits, dim=-1)

    while steps < total_steps:
        with torch.no_grad():
            logits = online(one_hot(s).unsqueeze(0))
            p = dist(logits)
            q = torch.sum(p * z.view(1, 1, -1), dim=-1)  # [1, A]
            a = int(torch.argmax(q, dim=-1).item())
        s2, r, done, _ = env.step(a)
        rb.push(s, a, r, s2, done)
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1))) if done else s2
        steps += 1

        if steps >= start_learning_after and len(rb) >= batch_size:
            S, A, R, S2, D = rb.sample(batch_size, rng)
            S_t = torch.stack([one_hot(i.item() if hasattr(i, 'item') else int(i)) for i in S.squeeze(-1)]).to(device)
            A_t = torch.from_numpy(A).long().to(device)
            R_t = torch.from_numpy(R).float().to(device)
            D_t = torch.from_numpy(D).float().to(device)
            S2_t = torch.stack([one_hot(i.item() if hasattr(i, 'item') else int(i)) for i in S2.squeeze(-1)]).to(device)

            with torch.no_grad():
                next_logits = target(S2_t)  # [B, A, K]
                next_p = dist(next_logits)
                next_q = torch.sum(next_p * z.view(1, 1, -1), dim=-1)  # [B, A]
                next_a = torch.argmax(next_q, dim=-1)  # [B]
                next_dist = next_p[torch.arange(batch_size), next_a]  # [B, K]
                Tz = R_t.view(-1, 1) + (1.0 - D_t.view(-1, 1)) * gamma * z.view(1, -1)
                Tz = Tz.clamp(v_min, v_max)
                b = (Tz - v_min) / delta_z
                l = b.floor().long()
                u = b.ceil().long()
                m = torch.zeros(batch_size, n_atoms, device=device)
                offset = torch.arange(batch_size, device=device).unsqueeze(1)
                m.index_add_(1, l.clamp(0, n_atoms - 1).view(-1), (next_dist * (u - b)).view(-1))
                m.index_add_(1, u.clamp(0, n_atoms - 1).view(-1), (next_dist * (b - l)).view(-1))
                m = m.view(batch_size, n_atoms)

            logits = online(S_t)  # [B, A, K]
            logits_a = logits[torch.arange(batch_size), A_t]  # [B, K]
            logp = log_softmax(logits_a)
            loss = -(m * logp).sum(dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
            opt.step()

        if steps % target_sync_interval == 0:
            target.load_state_dict(online.state_dict())

    # 导出贪心策略
    policy = []
    import torch
    with torch.no_grad():
        for s_idx in range(env.n_states):
            p = dist(online(one_hot(s_idx).unsqueeze(0)))
            q = torch.sum(p * z.view(1, 1, -1), dim=-1)
            a = int(torch.argmax(q, dim=-1).item())
            policy.append(a)

    return C51Result(policy=policy, iters=steps)

