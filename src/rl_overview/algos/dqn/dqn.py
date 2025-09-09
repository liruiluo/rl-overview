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
from typing import Tuple, Deque
from collections import deque

import numpy as np
from ...replay.buffer import ReplayBuffer
from ...replay.prioritized import PrioritizedReplayBuffer
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
    n_step: int = 1,
    prioritized_replay: bool = False,
    prio_alpha: float = 0.6,
    prio_beta: float = 0.4,
    prio_eps: float = 1e-3,
    noisy: bool = False,
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

    # 探测观测形态：离散（int -> one-hot）或向量（np.ndarray）
    s = env.reset(seed=seed)
    if isinstance(s, (int, np.integer)):
        discrete_obs = True
        obs_dim = getattr(env, "n_states", None)
        assert obs_dim is not None, "离散观测需提供 n_states"
    else:
        discrete_obs = False
        if hasattr(env, "obs_shape") and env.obs_shape is not None:
            obs_dim = int(env.obs_shape[0])
        else:
            obs_dim = int(np.asarray(s).shape[-1])
    n_actions = int(getattr(env, "n_actions"))
    if discrete_obs:
        from ...nn.dqn_net import build_tiny_qnet as _maker_small
        maker = build_dueling_qnet if dueling else _maker_small
    else:
        if noisy:
            from ...nn.dqn_net import build_noisy_qnet as _maker_mlp
        else:
            from ...nn.dqn_net import build_mlp_qnet as _maker_mlp
        maker = build_dueling_qnet if dueling else _maker_mlp
    online = maker((obs_dim,), n_actions).to(device)
    target = maker((obs_dim,), n_actions).to(device)
    target.load_state_dict(online.state_dict())
    optim_ = optim.Adam(online.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    use_prio = bool(prioritized_replay)
    if use_prio:
        rb = PrioritizedReplayBuffer(replay_capacity, alpha=float(prio_alpha), eps=float(prio_eps))
    else:
        rb = ReplayBuffer(replay_capacity)
    steps = 0
    n = max(1, int(n_step))
    trace: Deque[tuple] = deque(maxlen=n)

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
        # n-step accumulation
        trace.append((s, a, r))
        def flush_trace(s_next, done_flag):
            Rn = 0.0
            g = 1.0
            for (_, _, r_i) in trace:
                Rn += g * r_i
                g *= gamma
            s0, a0, _ = trace[0]
            if use_prio:
                rb.push(s0, a0, Rn, s_next, done_flag, priority=None)
            else:
                rb.push(s0, a0, Rn, s_next, done_flag)

        if len(trace) == n:
            flush_trace(s2, done)
            trace.popleft()
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1))) if done else s2
        steps += 1

        if steps >= start_learning_after and len(rb) >= batch_size:
            if use_prio:
                # 退火 beta
                cur_beta = float(prio_beta)
                try:
                    beta_final = float(prio_beta)
                except Exception:
                    beta_final = cur_beta
                frac = min(1.0, steps / max(1, total_steps))
                cur_beta = cur_beta + (beta_final - cur_beta) * frac
                S, A, R, S2, D, W, idxs = rb.sample(batch_size, beta=cur_beta, rng=rng)
                W_t = torch.from_numpy(W).float().to(device)
            else:
                S, A, R, S2, D = rb.sample(batch_size, rng)
            if discrete_obs:
                S_in = torch.from_numpy(_one_hot(obs_dim, S.squeeze(-1))).to(device)
                S2_in = torch.from_numpy(_one_hot(obs_dim, S2.squeeze(-1))).to(device)
            else:
                S_in = torch.from_numpy(S).float().to(device)
                S2_in = torch.from_numpy(S2).float().to(device)
            A_t = torch.from_numpy(A).long().to(device)
            R_t = torch.from_numpy(R).float().to(device)
            D_t = torch.from_numpy(D).float().to(device)

            q = online(S_in).gather(1, A_t.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                if double_dqn:
                    next_actions = torch.argmax(online(S2_in), dim=1)
                    next_q = target(S2_in).gather(1, next_actions.view(-1, 1)).squeeze(1)
                else:
                    next_q = torch.max(target(S2_in), dim=1).values
                target_q = R_t + (1.0 - D_t) * (gamma ** n) * next_q

            td_err = target_q - q
            if use_prio:
                loss = (W_t * (td_err ** 2)).mean()
            else:
                loss = loss_fn(q, target_q)
            optim_.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
            optim_.step()

            if use_prio:
                new_prios = (td_err.detach().abs().cpu().numpy() + prio_eps)
                rb.update_priorities(idxs, new_prios)

        if steps % target_sync_interval == 0:
            target.load_state_dict(online.state_dict())

    # 导出贪心策略（针对离散状态）
    policy = []
    with torch.no_grad():
        if discrete_obs:
            for s_idx in range(getattr(env, "n_states")):
                q = online(torch.from_numpy(_one_hot(obs_dim, np.array([s_idx]))).to(device))
                policy.append(int(torch.argmax(q, dim=-1).item()))
        else:
            # 对连续观测任务不导出全局策略，这里返回空策略占位
            policy = []

    return DQNResult(policy=policy, iters=steps)
