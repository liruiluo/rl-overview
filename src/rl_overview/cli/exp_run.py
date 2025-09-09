from __future__ import annotations

import hydra
from omegaconf import DictConfig
from pathlib import Path
from rich.console import Console

from ..train import _instantiate_env
from ..utils.logger import EpisodeLogger


console = Console()


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    env = _instantiate_env(cfg)
    algo = cfg.algo.name
    out_file = Path(getattr(cfg, "metrics_file", "metrics.csv"))
    logger = EpisodeLogger(out_file, print_every=max(1, int(getattr(cfg, "print_every", 0))))

    if algo == "q_learning":
        from ..algos.td.q_learning import q_learning
        q_learning(
            env=env,
            gamma=float(cfg.algo.gamma),
            alpha=float(cfg.algo.alpha),
            epsilon=float(cfg.algo.epsilon),
            episodes=int(cfg.algo.episodes),
            max_steps=int(cfg.algo.max_steps),
            seed=int(cfg.algo.seed),
            logger=logger,
        )
    elif algo == "sarsa":
        from ..algos.td.sarsa import sarsa
        sarsa(
            env=env,
            gamma=float(cfg.algo.gamma),
            alpha=float(cfg.algo.alpha),
            epsilon=float(cfg.algo.epsilon),
            episodes=int(cfg.algo.episodes),
            max_steps=int(cfg.algo.max_steps),
            seed=int(cfg.algo.seed),
            logger=logger,
        )
    elif algo == "sarsa_lambda":
        from ..algos.td.sarsa_lambda import sarsa_lambda
        sarsa_lambda(
            env=env,
            gamma=float(cfg.algo.gamma),
            lam=float(cfg.algo.lam),
            alpha=float(cfg.algo.alpha),
            epsilon=float(cfg.algo.epsilon),
            episodes=int(cfg.algo.episodes),
            max_steps=int(cfg.algo.max_steps),
            seed=int(cfg.algo.seed),
            logger=logger,
        )
    elif algo == "mc_control":
        from ..algos.mc.mc_control import mc_control_every_visit
        mc_control_every_visit(
            env=env,
            gamma=float(cfg.algo.gamma),
            epsilon=float(cfg.algo.epsilon),
            episodes=int(cfg.algo.episodes),
            max_steps=int(cfg.algo.max_steps),
            seed=int(cfg.algo.seed),
            logger=logger,
        )
    else:
        raise SystemExit("exp_run 目前支持: q_learning, sarsa, sarsa_lambda, mc_control")

    logger.close()
    console.rule("实验完成")
    console.print(f"指标已写入: {out_file}")


if __name__ == "__main__":
    main()

