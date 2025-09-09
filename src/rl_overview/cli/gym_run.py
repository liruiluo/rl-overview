from __future__ import annotations

import hydra
from omegaconf import DictConfig
from rich.console import Console

from ..envs.wrappers import gym_to_step_env
from ..algos.dqn.dqn import train_dqn
from ..algos.pg.ppo import train_ppo

console = Console()


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    env = gym_to_step_env(cfg.env.env_id)
    algo = cfg.algo.name
    if algo == "dqn":
        res = train_dqn(
            env=env,
            gamma=cfg.algo.gamma,
            lr=cfg.algo.lr,
            batch_size=cfg.algo.batch_size,
            replay_capacity=cfg.algo.replay_capacity,
            start_learning_after=cfg.algo.start_learning_after,
            target_sync_interval=cfg.algo.target_sync_interval,
            total_steps=cfg.algo.total_steps,
            epsilon_start=cfg.algo.epsilon_start,
            epsilon_end=cfg.algo.epsilon_end,
            epsilon_decay_steps=cfg.algo.epsilon_decay_steps,
            double_dqn=cfg.algo.double_dqn,
            dueling=cfg.algo.dueling,
            n_step=cfg.algo.n_step,
            prioritized_replay=cfg.algo.prioritized_replay,
            prio_alpha=cfg.algo.prio_alpha,
            prio_beta=cfg.algo.prio_beta,
            prio_eps=cfg.algo.prio_eps,
            seed=cfg.seed,
        )
        console.rule("Gym DQN 完成")
        console.print(res)
    elif algo == "ppo":
        res = train_ppo(
            env=env,
            gamma=cfg.algo.gamma,
            lam=cfg.algo.lam,
            lr=cfg.algo.lr,
            steps_per_epoch=cfg.algo.steps_per_epoch,
            epochs=cfg.algo.epochs,
            clip_ratio=cfg.algo.clip_ratio,
            train_iters=cfg.algo.train_iters,
            vf_coef=cfg.algo.vf_coef,
            ent_coef=cfg.algo.ent_coef,
            seed=cfg.seed,
        )
        console.rule("Gym PPO 完成")
        console.print(res)
    else:
        raise SystemExit("未知算法：支持 dqn 或 ppo")


if __name__ == "__main__":
    main()

