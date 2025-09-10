from __future__ import annotations

import hydra
from omegaconf import DictConfig
import numpy as np
from rich.console import Console

from ..envs.gridworld1d import GridWorld1D
from ..llm.prefs_numpy import rollouts, bag_of_states, pref_pairs_from_trajs, train_logistic_pairwise, predict_pref_score

console = Console()


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # 使用 GridWorld1D 生成偏好数据（可扩展到其他离散环境）
    env = GridWorld1D(
        n_states=cfg.env.n_states,
        step_penalty=float(cfg.env.step_penalty),
        right_terminal_reward=float(cfg.env.right_terminal_reward),
    )
    # 用简单的向右贪心策略生成轨迹（也可混入随机政策）
    def pi_right(s: int) -> int:
        return 1 if s not in env.terminal_states else 0

    trajs = rollouts(env, policy=pi_right, episodes=int(cfg.get("episodes", 200)), max_steps=200, seed=int(cfg.seed))
    X = np.stack([bag_of_states(t, env.n_states) for t in trajs])
    pairs = pref_pairs_from_trajs(trajs, noise_std=float(cfg.get("noise_std", 0.0)), seed=int(cfg.seed))
    w = train_logistic_pairwise(X, pairs, lr=float(cfg.get("lr", 1e-2)), epochs=int(cfg.get("epochs", 500)), l2=float(cfg.get("l2", 1e-3)), seed=int(cfg.seed))

    # 评估：随机抽取若干对做偏好方向判断
    rng = np.random.default_rng(cfg.seed)
    acc_cnt = 0
    tot = 0
    for _ in range(100):
        i = int(rng.integers(0, len(trajs) - 1))
        j = i + 1
        r_i = sum(trajs[i].rewards)
        r_j = sum(trajs[j].rewards)
        y = 1 if r_i >= r_j else 0
        pred = 1 if predict_pref_score(w, X[i] - X[j]) >= 0 else 0
        acc_cnt += int(pred == y)
        tot += 1

    console.rule("偏好模型（Numpy Logistic）")
    console.print({
        "pairs": len(pairs),
        "acc_estimate": acc_cnt / max(1, tot)
    })


if __name__ == "__main__":
    main()

