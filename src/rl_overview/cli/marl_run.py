from __future__ import annotations

import hydra
from omegaconf import DictConfig
import numpy as np
from rich.console import Console

from ..marl.envs.normal_form import NormalFormGame
from ..marl.independent_q import independent_q

console = Console()


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.env.name != "matching_pennies":
        raise SystemExit("当前MARL演示仅支持 env=normal_form_matching_pennies 作为示例")

    payoff = np.array(cfg.env.payoff, dtype=np.float64)
    game = NormalFormGame(payoff)

    Q1, Q2 = independent_q(
        game,
        alpha=0.1,
        epsilon=0.1,
        episodes=50_000,
        seed=cfg.seed,
    )
    console.rule("MARL: Matching Pennies 独立Q 学习结束")
    console.print("Q1:\n", Q1)
    console.print("Q2:\n", Q2)


if __name__ == "__main__":
    main()

