from __future__ import annotations

import hydra
from omegaconf import DictConfig
import numpy as np
from rich.console import Console

from ..bandits.bernoulli import BernoulliBandit
from ..bandits.ucb import ucb1
from ..bandits.ts import thompson_sampling_bernoulli

console = Console()


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    probs = np.array(cfg.env.probs, dtype=float)
    bandit = BernoulliBandit(probs)
    algo = cfg.algo.name
    if algo == "ucb1":
        res = ucb1(bandit, rounds=cfg.algo.rounds, c=cfg.algo.c, seed=cfg.seed)
    elif algo == "ts":
        res = thompson_sampling_bernoulli(bandit, rounds=cfg.algo.rounds, seed=cfg.seed)
    else:
        raise SystemExit("未知 bandit 算法：支持 ucb1, ts")
    console.rule(f"Bandit {algo} 结果")
    console.print(res)


if __name__ == "__main__":
    main()

