from __future__ import annotations

"""
最小 A2C（同步 Advantage Actor-Critic）。
支持离散状态(one-hot)/向量观测，离散动作。
"""

from dataclasses import dataclass
import numpy as np

from ...nn.torch_utils import is_torch_available, get_device


@dataclass
class A2CResult:
    policy: list[int]
    iters: int


def train_a2c(
    env,
    gamma: float = 0.99,
    lr: float = 3e-4,
    steps_per_update: int = 256,
    updates: int = 50,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    seed: int = 42,
) -> A2CResult:
    if not is_torch_available():
        raise RuntimeError("未安装 torch。`uv sync --extra torch` 后再运行 A2C。")

    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = get_device()
    rng = np.random.default_rng(seed)

    s0 = env.reset(seed=seed)
    if isinstance(s0, (int, np.integer)):
        discrete_obs = True
        S = int(getattr(env, "n_states"))
    else:
        discrete_obs = False
        if hasattr(env, "obs_shape") and env.obs_shape is not None:
            S = int(env.obs_shape[0])
        else:
            S = int(np.asarray(s0).shape[-1])
    A = int(getattr(env, "n_actions"))

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(S, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
            self.pi_head = nn.Linear(128, A)
            self.v_head = nn.Linear(128, 1)

        def pi(self, x):
            logits = self.pi_head(self.body(x))
            return torch.distributions.Categorical(logits=logits)

        def v(self, x):
            return self.v_head(self.body(x)).squeeze(-1)

    ac = ActorCritic().to(device)
    optim_ = optim.Adam(ac.parameters(), lr=lr)

    def obs_tensor(s):
        if discrete_obs:
            x = torch.zeros(S, dtype=torch.float32, device=device)
            x[int(s)] = 1.0
            return x
        else:
            return torch.tensor(np.asarray(s), dtype=torch.float32, device=device)

    steps = 0
    s = s0
    for upd in range(updates):
        obs_l, act_l, rew_l, logp_l, val_l, done_l = [], [], [], [], [], []
        for t in range(steps_per_update):
            with torch.no_grad():
                pi = ac.pi(obs_tensor(s))
                v = ac.v(obs_tensor(s))
            a = int(pi.sample().item())
            logp = float(pi.log_prob(torch.tensor(a, device=device)).item())
            s2, r, done, _ = env.step(a)
            obs_l.append(s)
            act_l.append(a)
            rew_l.append(float(r))
            logp_l.append(logp)
            val_l.append(float(v.item()))
            done_l.append(bool(done))
            s = env.reset(seed=int(rng.integers(0, 2**31 - 1))) if done else s2
            steps += 1

        with torch.no_grad():
            v_last = ac.v(obs_tensor(s)).item()
        vals = np.array(val_l + [v_last], dtype=np.float32)
        rews = np.array(rew_l, dtype=np.float32)
        dones = np.array(done_l, dtype=np.bool_)

        # 一步优势（TD误差的回溯形式，带终止修正）
        adv = np.zeros_like(rews)
        last_adv = 0.0
        for t in reversed(range(len(rews))):
            nonterminal = 1.0 - float(dones[t])
            delta = rews[t] + gamma * vals[t + 1] * nonterminal - vals[t]
            adv[t] = delta + gamma * nonterminal * last_adv
            last_adv = adv[t]
        ret = adv + vals[:-1]

        obs_t = torch.stack([obs_tensor(o) for o in obs_l]) if discrete_obs else torch.tensor(np.asarray(obs_l), dtype=torch.float32, device=device)
        act_t = torch.tensor(act_l, dtype=torch.int64, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)
        logp_old_t = torch.tensor(logp_l, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        pi = ac.pi(obs_t)
        logp = pi.log_prob(act_t)
        ent = pi.entropy().mean()
        v = ac.v(obs_t)

        pi_loss = -(logp * adv_t).mean() - ent_coef * ent
        v_loss = ((v - ret_t) ** 2).mean() * vf_coef
        loss = pi_loss + v_loss

        optim_.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ac.parameters(), max_norm=0.5)
        optim_.step()

    # 导出离散策略
    policy = []
    import torch
    with torch.no_grad():
        if discrete_obs:
            for s_idx in range(S):
                a = int(torch.argmax(ac.pi(obs_tensor(s_idx)).logits).item())
                policy.append(a)

    return A2CResult(policy=policy, iters=steps)

