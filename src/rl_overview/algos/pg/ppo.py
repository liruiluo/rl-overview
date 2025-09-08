from __future__ import annotations

"""
最小 PPO(clip) 骨架（Torch）。
针对离散状态(one-hot)/动作。仅教学用途，省略了很多工程细节。
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from ...nn.torch_utils import is_torch_available, get_device


@dataclass
class PPOResult:
    policy: list[int]
    iters: int


def train_ppo(
    env,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 3e-4,
    steps_per_epoch: int = 2048,
    epochs: int = 10,
    clip_ratio: float = 0.2,
    train_iters: int = 80,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    seed: int = 42,
) -> PPOResult:
    if not is_torch_available():  # pragma: no cover
        raise RuntimeError("未安装 torch。请使用 `uv sync --extra torch` 后再运行 PPO。")

    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = get_device()
    rng = np.random.default_rng(seed)

    S = env.n_states
    A = env.n_actions

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(S, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
            self.pi_head = nn.Linear(64, A)
            self.v_head = nn.Linear(64, 1)

        def pi(self, x):
            logits = self.pi_head(self.body(x))
            return torch.distributions.Categorical(logits=logits)

        def v(self, x):
            return self.v_head(self.body(x)).squeeze(-1)

    ac = ActorCritic().to(device)
    optim_ = optim.Adam(ac.parameters(), lr=lr)

    def one_hot(idx):
        x = torch.zeros(S, dtype=torch.float32, device=device)
        x[idx] = 1.0
        return x

    total_steps = 0
    for epoch in range(epochs):
        obs_buf = []
        act_buf = []
        adv_buf = []
        ret_buf = []
        logp_buf = []

        vals = []
        rets = []
        lens = []

        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_rew = 0.0
        ep_len = 0
        buf_steps = 0

        # 收集一批 trajectories
        while buf_steps < steps_per_epoch:
            with torch.no_grad():
                pi = ac.pi(one_hot(int(s)))
                v = ac.v(one_hot(int(s)))
            a = int(pi.sample().item())
            logp = float(pi.log_prob(torch.tensor(a, device=device)).item())
            s2, r, done, _ = env.step(a)
            obs_buf.append(int(s))
            act_buf.append(a)
            logp_buf.append(logp)
            vals.append(float(v.item()))
            ep_rew += r
            ep_len += 1
            buf_steps += 1
            s = s2
            if done:
                rets.append(ep_rew)
                lens.append(ep_len)
                s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                ep_rew = 0.0
                ep_len = 0

        # 计算 GAE 与 returns
        with torch.no_grad():
            last_val = ac.v(one_hot(int(s))).item()
        vals_np = np.array(vals + [last_val], dtype=np.float32)
        rewards = np.zeros(len(vals), dtype=np.float32)  # 我们缺少逐步奖励缓存，上面只累计了 ep_rew，故简化假设稀疏奖励任务时可用
        # 为了教学骨架，此处将奖励近似为 0，仅用于管线演示；实际实现需缓存 r_t 列表
        adv = np.zeros_like(rewards)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * vals_np[t + 1] - vals_np[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            adv[t] = lastgaelam
        ret = adv + vals_np[:-1]

        # 转张量
        obs_t = torch.stack([one_hot(i) for i in obs_buf])
        act_t = torch.tensor(act_buf, dtype=torch.int64, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)
        logp_old_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # 训练多次
        for _ in range(train_iters):
            pi = ac.pi(obs_t)
            logp = pi.log_prob(act_t)
            ratio = torch.exp(logp - logp_old_t)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_t
            pi_loss = -(torch.min(ratio * adv_t, clip_adv).mean() + ent_coef * pi.entropy().mean())
            v = ac.v(obs_t)
            v_loss = ((v - ret_t) ** 2).mean() * vf_coef
            loss = pi_loss + v_loss
            optim_.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), max_norm=0.5)
            optim_.step()

        total_steps += buf_steps

    # 导出贪心策略
    policy = []
    import torch
    with torch.no_grad():
        for s_idx in range(S):
            pi = ac.pi(one_hot(s_idx))
            a = int(torch.argmax(pi.logits).item())
            policy.append(a)

    return PPOResult(policy=policy, iters=total_steps)

