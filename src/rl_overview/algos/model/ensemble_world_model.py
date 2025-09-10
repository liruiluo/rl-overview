from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
from ...nn.torch_utils import is_torch_available, get_device


@dataclass
class DiscreteEnsembleModel:
    n_states: int
    n_actions: int
    nets: list
    device: object

    def predict(self, s: int, a: int):
        import torch
        probs = []
        rews = []
        for net in self.nets:
            with torch.no_grad():
                x = torch.tensor([[s, a]], dtype=torch.long, device=self.device)
                logits, r = net(x)
                p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                probs.append(p)
                rews.append(float(r.item()))
        probs = np.stack(probs, axis=0)
        rews = np.array(rews, dtype=np.float64)
        return probs, rews


def fit_ensemble_discrete(env, episodes: int = 200, max_steps: int = 200, n_models: int = 5, lr: float = 1e-3, seed: int = 42) -> DiscreteEnsembleModel:
    if not is_torch_available():
        raise RuntimeError("未安装 torch。`uv sync --extra torch` 后再训练 ensemble 模型。")
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
            self.emb_s = nn.Embedding(S, 16)
            self.emb_a = nn.Embedding(A, 8)
            self.body = nn.Sequential(nn.Linear(24, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.head_logits = nn.Linear(64, S)
            self.head_r = nn.Linear(64, 1)
        def forward(self, x):
            s_idx = x[:, 0]
            a_idx = x[:, 1]
            h = torch.cat([self.emb_s(s_idx), self.emb_a(a_idx)], dim=-1)
            h = self.body(h)
            return self.head_logits(h), self.head_r(h).squeeze(-1)

    nets = [Net().to(device) for _ in range(n_models)]
    opts = [optim.Adam(n.parameters(), lr=lr) for n in nets]
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    bs = 256
    for m, (net, opt) in enumerate(zip(nets, opts)):
        for epoch in range(10):
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

    return DiscreteEnsembleModel(n_states=S, n_actions=A, nets=nets, device=device)


def pets_action_discrete(model: DiscreteEnsembleModel, s0: int, horizon: int = 8, n_candidates: int = 200, gamma: float = 0.99, risk_coef: float = 0.5, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    best_G = -1e9
    best_a0 = 0
    for _ in range(n_candidates):
        a0 = int(rng.integers(0, model.n_actions))
        Gs = []
        for m in range(len(model.nets)):
            s = s0
            G = 0.0
            g = 1.0
            for t in range(horizon):
                a = a0 if t == 0 else int(rng.integers(0, model.n_actions))
                probs, rews = model.predict(s, a)
                p_m = probs[m]
                s2 = int(rng.choice(np.arange(model.n_states), p=p_m / p_m.sum()))
                r = float(rews[m])
                G += g * r
                g *= gamma
                s = s2
            Gs.append(G)
        meanG = float(np.mean(Gs))
        stdG = float(np.std(Gs))
        score = meanG - risk_coef * stdG
        if score > best_G:
            best_G = score
            best_a0 = a0
    return best_a0

