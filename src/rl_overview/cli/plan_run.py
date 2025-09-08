from __future__ import annotations

import hydra
from omegaconf import DictConfig
from rich.console import Console

from ..envs.gridworld1d import GridWorld1D
from ..algos.planning.mcts import mcts_action, random_shooting_action

console = Console()


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    env = GridWorld1D(
        n_states=cfg.env.n_states,
        step_penalty=float(cfg.env.step_penalty),
        right_terminal_reward=float(cfg.env.right_terminal_reward),
    )
    s0 = (env.n_states - 1) // 2
    if cfg.algo.name == "mcts":
        a = mcts_action(env, s0, gamma=cfg.algo.gamma, num_simulations=cfg.algo.num_simulations, max_depth=cfg.algo.max_depth)
        console.rule("MCTS 规划")
        console.print(f"在状态 {s0} 选择动作: {a}")
    elif cfg.algo.name == "shooting":
        a = random_shooting_action(env, s0, horizon=cfg.algo.horizon, n_candidates=cfg.algo.n_candidates, gamma=cfg.algo.gamma)
        console.rule("随机射击规划")
        console.print(f"在状态 {s0} 选择动作: {a}")
    else:
        raise SystemExit("未知规划算法：支持 mcts / shooting")


if __name__ == "__main__":
    main()

