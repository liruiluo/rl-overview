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
    minibatch_size: int | None = None,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    target_kl: float | None = None,
    anneal_lr: bool = False,
    seed: int = 42,
) -> PPOResult:
    if not is_torch_available():  # pragma: no cover
        raise RuntimeError("未安装 torch。请使用 `uv sync --extra torch` 后再运行 PPO。")

    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = get_device()
    rng = np.random.default_rng(seed)

    # 观测/动作自适应
    s_probe = env.reset(seed=seed)
    if isinstance(s_probe, (int, np.integer)):
        discrete_obs = True
        S = int(getattr(env, "n_states"))
    else:
        discrete_obs = False
        if hasattr(env, "obs_shape") and env.obs_shape is not None:
            S = int(env.obs_shape[0])
        else:
            S = int(np.asarray(s_probe).shape[-1])
    A = int(getattr(env, "n_actions"))

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

    def obs_tensor(s):
        if discrete_obs:
            x = torch.zeros(S, dtype=torch.float32, device=device)
            x[int(s)] = 1.0
            return x
        else:
            return torch.tensor(np.asarray(s), dtype=torch.float32, device=device)

    total_steps = 0
    for epoch in range(epochs):
        obs_buf = []        # s_t (int)
        act_buf = []        # a_t
        rew_buf = []        # r_t
        val_buf = []        # V(s_t)
        logp_buf = []       # log π(a_t|s_t)
        done_buf = []       # done_t

        s = s_probe
        buf_steps = 0

        # 收集一批 trajectories（on-policy）
        while buf_steps < steps_per_epoch:
            with torch.no_grad():
                pi_dist = ac.pi(obs_tensor(s))
                v_t = ac.v(obs_tensor(s))
            a = int(pi_dist.sample().item())
            logp = float(pi_dist.log_prob(torch.tensor(a, device=device)).item())
            s2, r, done, _ = env.step(a)

            obs_buf.append(int(s))
            act_buf.append(a)
            rew_buf.append(float(r))
            val_buf.append(float(v_t.item()))
            logp_buf.append(logp)
            done_buf.append(bool(done))

            s = env.reset(seed=int(rng.integers(0, 2**31 - 1))) if done else s2
            buf_steps += 1

        # 计算 GAE 与 returns（按批次，忽略跨批次的 bootstrapping）
        with torch.no_grad():
            last_val = ac.v(obs_tensor(s)).item()
        vals_np = np.array(val_buf + [last_val], dtype=np.float32)
        rews_np = np.array(rew_buf, dtype=np.float32)
        dones_np = np.array(done_buf, dtype=np.bool_)

        adv = np.zeros_like(rews_np)
        lastgaelam = 0.0
        for t in reversed(range(len(rews_np))):
            nonterminal = 1.0 - float(dones_np[t])
            delta = rews_np[t] + gamma * vals_np[t + 1] * nonterminal - vals_np[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + vals_np[:-1]

        # 转张量
        if discrete_obs:
            obs_t = torch.stack([obs_tensor(i) for i in obs_buf])
        else:
            obs_t = torch.tensor(np.asarray(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(act_buf, dtype=torch.int64, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)
        logp_old_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # 训练多次（支持 minibatch）
        N = obs_t.shape[0]
        mb_size = int(minibatch_size or N)
        for it in range(train_iters):
            perm = torch.randperm(N, device=device)
            for i in range(0, N, mb_size):
                idx = perm[i : i + mb_size]
                mb_obs = obs_t[idx]
                mb_act = act_t[idx]
                mb_adv = adv_t[idx]
                mb_ret = ret_t[idx]
                mb_logp_old = logp_old_t[idx]

                pi = ac.pi(mb_obs)
                logp = pi.log_prob(mb_act)
                ratio = torch.exp(logp - mb_logp_old)
                clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_adv
                pi_loss = -(torch.min(ratio * mb_adv, clip_adv).mean() + ent_coef * pi.entropy().mean())
                v = ac.v(mb_obs)
                v_loss = ((v - mb_ret) ** 2).mean() * vf_coef
                loss = pi_loss + v_loss
                optim_.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), max_norm=0.5)
                optim_.step()

            # KL 早停（估计为均值 KL）
            if target_kl is not None:
                with torch.no_grad():
                    pi = ac.pi(obs_t)
                    logp = pi.log_prob(act_t)
                    approx_kl = (logp_old_t - logp).mean().item()
                if approx_kl > target_kl:
                    break

        # 学习率退火
        if anneal_lr:
            frac = 1.0 - (epoch + 1) / max(1, epochs)
            for g in optim_.param_groups:
                g["lr"] = lr * frac

        total_steps += buf_steps

    # 导出贪心策略
    policy = []
    import torch
    with torch.no_grad():
        if discrete_obs:
            for s_idx in range(S):
                pi = ac.pi(obs_tensor(s_idx))
                a = int(torch.argmax(pi.logits).item())
                policy.append(a)

    return PPOResult(policy=policy, iters=total_steps)
