from __future__ import annotations

"""
学习式世界模型（离散状态）：
- 通过采样经验拟合 p(s'|s,a) 与 r(s,a,s') 的期望
"""

from dataclasses import dataclass
import numpy as np
from ...nn.torch_utils import is_torch_available, get_device


@dataclass
class LearnedDiscreteModel:
    n_states: int
    n_actions: int
    net: object  # torch.nn.Module
    device: object

    def predict_probs(self, s: int, a: int):
        import torch
        x = torch.tensor([s, a], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, r = self.net(x.unsqueeze(0))
            p = torch.softmax(logits, dim=-1).squeeze(0)
        return p.cpu().numpy()

    def predict_reward(self, s: int, a: int) -> float:
        import torch
        x = torch.tensor([s, a], dtype=torch.long, device=self.device)
        with torch.no_grad():
            _logits, r = self.net(x.unsqueeze(0))
        return float(r.item())


def fit_discrete_world_model(env, episodes: int = 200, max_steps: int = 200, lr: float = 1e-3, seed: int = 42) -> LearnedDiscreteModel:
    if not is_torch_available():
        raise RuntimeError("未安装 torch。`uv sync --extra torch` 后再训练世界模型。")
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = get_device()
    rng = np.random.default_rng(seed)
    S = int(getattr(env, "n_states"))
    A = int(getattr(env, "n_actions"))

    data = []
    for ep in range(episodes):
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(max_steps):
            a = int(rng.integers(0, A))
            s2, r, done, _ = env.step(a)
            data.append((s, a, s2, r))
            s = s2
            if done:
                break

    X = torch.tensor([[s, a] for (s, a, _s2, _r) in data], dtype=torch.long, device=device)
    y_s2 = torch.tensor([s2 for (_s, _a, s2, _r) in data], dtype=torch.long, device=device)
    y_r = torch.tensor([r for (_s, _a, _s2, r) in data], dtype=torch.float32, device=device)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb_s = nn.Embedding(S, 32)
            self.emb_a = nn.Embedding(A, 16)
            self.body = nn.Sequential(nn.Linear(32 + 16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.head_logits = nn.Linear(64, S)
            self.head_r = nn.Linear(64, 1)

        def forward(self, x):  # x: [B, 2], longs
            s_idx = x[:, 0]
            a_idx = x[:, 1]
            h = torch.cat([self.emb_s(s_idx), self.emb_a(a_idx)], dim=-1)
            h = self.body(h)
            return self.head_logits(h), self.head_r(h).squeeze(-1)

    net = Net().to(device)
    opt = optim.Adam(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    bs = 256
    for epoch in range(20):
        perm = torch.randperm(X.shape[0], device=device)
        for i in range(0, X.shape[0], bs):
            idx = perm[i:i+bs]
            xb = X[idx]
            sb = y_s2[idx]
            rb = y_r[idx]
            logits, r_pred = net(xb)
            loss = ce(logits, sb) + 0.1 * mse(r_pred, rb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()

    return LearnedDiscreteModel(n_states=S, n_actions=A, net=net, device=device)


def plan_with_learned_model(model: LearnedDiscreteModel, s0: int, horizon: int = 10, n_candidates: int = 100, gamma: float = 0.99, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    best_G = -1e9
    best_a0 = 0
    for _ in range(n_candidates):
        s = s0
        G = 0.0
        g = 1.0
        a0 = int(rng.integers(0, model.n_actions))
        for t in range(horizon):
            a = a0 if t == 0 else int(rng.integers(0, model.n_actions))
            probs = model.predict_probs(s, a)
            s2 = int(rng.choice(np.arange(model.n_states), p=probs/probs.sum()))
            r = model.predict_reward(s, a)
            G += g * r
            g *= gamma
            s = s2
        if G > best_G:
            best_G = G
            best_a0 = a0
    return best_a0

