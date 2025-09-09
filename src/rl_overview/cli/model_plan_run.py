from __future__ import annotations

import hydra
from omegaconf import DictConfig
from rich.console import Console

from ..envs.gridworld1d import GridWorld1D
from ..algos.model.world_model import fit_discrete_world_model, plan_with_learned_model

console = Console()


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    env = GridWorld1D(
        n_states=cfg.env.n_states,
        step_penalty=float(cfg.env.step_penalty),
        right_terminal_reward=float(cfg.env.right_terminal_reward),
    )
    model = fit_discrete_world_model(env, episodes=cfg.get("episodes", 200), max_steps=cfg.get("max_steps", 200), lr=1e-3, seed=cfg.seed)
    s0 = (env.n_states - 1) // 2
    a = plan_with_learned_model(model, s0, horizon=10, n_candidates=200, gamma=0.99, seed=cfg.seed)
    console.rule("Learned World Model 规划")
    console.print(f"在状态 {s0} 选择动作: {a}")


if __name__ == "__main__":
    main()

