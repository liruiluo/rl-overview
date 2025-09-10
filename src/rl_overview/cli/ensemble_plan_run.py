from __future__ import annotations

import hydra
from omegaconf import DictConfig
from rich.console import Console

from ..envs.gridworld1d import GridWorld1D
from ..algos.model.ensemble_world_model import fit_ensemble_discrete, pets_action_discrete

console = Console()


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    env = GridWorld1D(
        n_states=cfg.env.n_states,
        step_penalty=float(cfg.env.step_penalty),
        right_terminal_reward=float(cfg.env.right_terminal_reward),
    )
    model = fit_ensemble_discrete(env, episodes=cfg.get("episodes", 200), max_steps=cfg.get("max_steps", 200), n_models=cfg.get("n_models", 5), lr=1e-3, seed=cfg.seed)
    s0 = (env.n_states - 1) // 2
    a = pets_action_discrete(model, s0, horizon=cfg.get("horizon", 8), n_candidates=cfg.get("n_candidates", 200), gamma=cfg.get("gamma", 0.99), risk_coef=cfg.get("risk_coef", 0.5), seed=cfg.seed)
    console.rule("Ensemble PETS 规划")
    console.print(f"在状态 {s0} 选择动作: {a}")


if __name__ == "__main__":
    main()

