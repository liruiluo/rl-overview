from __future__ import annotations

import itertools
from pathlib import Path
import hydra
from omegaconf import DictConfig
from rich.console import Console

from ..train import _instantiate_env
from ..utils.logger import EpisodeLogger
from ..algos.td.q_learning import q_learning

console = Console()


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    env_base = _instantiate_env(cfg)
    eps_list = list(getattr(cfg, "eps_list", [0.05, 0.1, 0.2]))
    alpha_list = list(getattr(cfg, "alpha_list", [0.05, 0.1]))
    episodes = int(getattr(cfg, "episodes", 500))
    max_steps = int(getattr(cfg, "max_steps", 1000))
    seed = int(getattr(cfg, "seed", 42))

    outdir = Path("sweep")
    outdir.mkdir(exist_ok=True)

    for eps, alpha in itertools.product(eps_list, alpha_list):
        # 每个组合复用环境实例可能影响随机性，这里重新实例化
        env = _instantiate_env(cfg)
        sub = outdir / f"eps_{eps}_alpha_{alpha}"
        sub.mkdir(parents=True, exist_ok=True)
        logger = EpisodeLogger(sub / "metrics.csv", print_every=max(1, episodes // 10))
        console.print(f"运行 Q-learning: eps={eps} alpha={alpha}")
        q_learning(
            env=env,
            gamma=float(cfg.algo.gamma),
            alpha=float(alpha),
            epsilon=float(eps),
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            logger=logger,
        )
        logger.close()

    console.rule("网格搜索完成")


if __name__ == "__main__":
    main()

