from __future__ import annotations

import itertools
from pathlib import Path
import hydra
from omegaconf import DictConfig
from rich.console import Console

from ..envs.wrappers import gym_to_step_env
from ..algos.dqn.dqn import train_dqn
from ..utils.logger import EpisodeLogger

console = Console()


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # 参数网格
    doubles = list(getattr(cfg, "double_list", [True, False]))
    duelings = list(getattr(cfg, "dueling_list", [True, False]))
    prios = list(getattr(cfg, "prioritized_list", [True, False]))
    n_steps = list(getattr(cfg, "n_step_list", [1, 3]))
    noisys = list(getattr(cfg, "noisy_list", [False, True]))
    seeds = list(getattr(cfg, "seeds", [42]))

    total_steps = int(getattr(cfg, "total_steps", 50000))
    eval_interval = int(getattr(cfg, "eval_interval", 2000))
    eval_episodes = int(getattr(cfg, "eval_episodes", 5))

    outdir = Path(getattr(cfg, "outdir", "sweep_gym_dqn"))
    outdir.mkdir(parents=True, exist_ok=True)

    for dbl, due, prio, ns, noisy, seed in itertools.product(doubles, duelings, prios, n_steps, noisys, seeds):
        tag = f"double_{int(dbl)}__dueling_{int(due)}__prio_{int(prio)}__nstep_{ns}__noisy_{int(noisy)}__seed_{seed}"
        sub = outdir / tag
        sub.mkdir(parents=True, exist_ok=True)
        metrics_file = sub / "metrics.csv"
        logger = EpisodeLogger(metrics_file, print_every=max(1, total_steps // 10))
        console.print(f"[sweep] {tag}")

        env = gym_to_step_env(cfg.env.env_id)
        eval_env = gym_to_step_env(cfg.env.env_id)
        train_dqn(
            env=env,
            gamma=cfg.algo.gamma,
            lr=cfg.algo.lr,
            batch_size=cfg.algo.batch_size,
            replay_capacity=cfg.algo.replay_capacity,
            start_learning_after=cfg.algo.start_learning_after,
            target_sync_interval=cfg.algo.target_sync_interval,
            total_steps=total_steps,
            epsilon_start=cfg.algo.epsilon_start,
            epsilon_end=cfg.algo.epsilon_end,
            epsilon_decay_steps=cfg.algo.epsilon_decay_steps,
            double_dqn=bool(dbl),
            dueling=bool(due),
            n_step=int(ns),
            prioritized_replay=bool(prio),
            prio_alpha=cfg.algo.prio_alpha,
            prio_beta_start=getattr(cfg.algo, "prio_beta_start", 0.4),
            prio_beta_end=getattr(cfg.algo, "prio_beta_end", 1.0),
            prio_eps=cfg.algo.prio_eps,
            noisy=bool(noisy),
            eval_env=eval_env,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            logger=logger,
            seed=int(seed),
        )
        logger.close()

    console.rule("Gym DQN 参数网格完成")


if __name__ == "__main__":
    main()

