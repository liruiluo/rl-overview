from __future__ import annotations

import hydra
from omegaconf import DictConfig
from rich.console import Console
from pathlib import Path

from ..envs.rlvr_binary import BinaryParityEnv
from ..algos.pg.reinforce import train_reinforce
from ..utils.logger import EpisodeLogger

console = Console()


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    env = BinaryParityEnv(n_bits=cfg.env.n_bits, target_parity=cfg.env.target_parity)
    metrics_file = getattr(cfg, "metrics_file", None)
    logger = EpisodeLogger(Path(metrics_file)) if metrics_file else None
    res = train_reinforce(
        env=env,
        gamma=cfg.algo.gamma,
        lr=cfg.algo.lr,
        episodes=cfg.algo.episodes,
        max_steps=cfg.algo.max_steps,
        seed=cfg.seed,
        logger=logger,
    )
    console.rule("RLVR REINFORCE 完成")
    console.print(res)


if __name__ == "__main__":
    main()

