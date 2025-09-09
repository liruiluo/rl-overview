from __future__ import annotations

"""
MBPO (最小离散版)：
- 使用神经网络学习 p(s'|s,a) 与 r(s,a)（见 world_model.py）
- 交替从真实环境采样与从模型滚动生成短轨迹，使用 Q-learning 更新
"""

from dataclasses import dataclass
import numpy as np
from rich.console import Console

from .world_model import fit_discrete_world_model

console = Console()


@dataclass
class MBPOResult:
    policy: list[int]
    iters: int


def mbpo_discrete(
    env,
    gamma: float = 0.99,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    real_steps: int = 2000,
    model_train_episodes: int = 200,
    model_rollouts_per_update: int = 1000,
    model_horizon: int = 5,
    seed: int = 42,
):
    S = int(getattr(env, "n_states"))
    A = int(getattr(env, "n_actions"))
    Q = np.zeros((S, A), dtype=np.float64)
    rng = np.random.default_rng(seed)

    # 初始收集真实数据并训练世界模型
    model = fit_discrete_world_model(env, episodes=model_train_episodes, max_steps=200, lr=1e-3, seed=seed)

    s = env.reset(seed=seed)
    for t in range(1, real_steps + 1):
        # 真实环境一步
        a = int(rng.integers(0, A)) if rng.random() < epsilon else int(np.argmax(Q[s]))
        s2, r, done, _ = env.step(a)
        best_next = 0.0 if done else np.max(Q[s2])
        Q[s, a] += alpha * (r + gamma * best_next - Q[s, a])
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1))) if done else s2

        # 周期性用模型进行短滚动更新（数据增强）
        if t % 100 == 0:
            for _ in range(model_rollouts_per_update):
                s_model = int(rng.integers(0, S))
                done_m = s_model in getattr(env, "terminal_states")
                for h in range(model_horizon):
                    if done_m:
                        break
                    a_m = int(rng.integers(0, A)) if rng.random() < epsilon else int(np.argmax(Q[s_model]))
                    # 用模型采样一步
                    probs = model.predict_probs(s_model, a_m)
                    s2_m = int(rng.choice(np.arange(S), p=probs / probs.sum()))
                    r_m = model.predict_reward(s_model, a_m)
                    best_next_m = 0.0 if (s2_m in getattr(env, "terminal_states")) else np.max(Q[s2_m])
                    Q[s_model, a_m] += alpha * (r_m + gamma * best_next_m - Q[s_model, a_m])
                    s_model = s2_m
                    done_m = s_model in getattr(env, "terminal_states")
            console.log(f"MBPO: 模型滚动增强完成 @ step={t}")

    policy = [int(np.argmax(Q[s])) for s in range(S)]
    return MBPOResult(policy=policy, iters=real_steps)

