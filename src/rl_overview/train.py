from __future__ import annotations

import sys
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

console = Console()


@dataclass
class RunResult:
    policy: list[int]
    values: list[float]
    iters: int


def _instantiate_env(cfg: DictConfig):
    """
    根据 cfg.env.name 实例化环境。

    这里尽量保持“代码即讲解”：
    - 环境接口简单直接，便于后续替换/扩展；
    - 不依赖外部 Gym，以便我们在每个章节中逐步引入所需概念。
    """
    name = cfg.env.name
    if name == "gridworld1d":
        from .envs.gridworld1d import GridWorld1D

        return GridWorld1D(
            n_states=cfg.env.n_states,
            step_penalty=float(cfg.env.step_penalty),
            right_terminal_reward=float(cfg.env.right_terminal_reward),
        )
    if name == "rlvr_binary":
        from .envs.rlvr_binary import BinaryParityEnv

        return BinaryParityEnv(
            n_bits=int(cfg.env.n_bits),
            target_parity=int(cfg.env.target_parity),
        )
    if name == "gym":
        from .envs.wrappers import gym_to_step_env
        return gym_to_step_env(cfg.env.env_id)
    raise ValueError(f"未知环境: {name}")


def _run_algo(cfg: DictConfig, env) -> RunResult:
    name = cfg.algo.name
    if name == "value_iteration":
        from .algos.value.value_iteration import value_iteration

        policy, values, iters = value_iteration(
            env=env,
            gamma=float(cfg.algo.gamma),
            theta=float(cfg.algo.theta),
            max_iters=int(cfg.algo.max_iters),
        )
        return RunResult(policy=policy, values=values, iters=iters)
    if name == "policy_iteration":
        from .algos.value.policy_iteration import policy_iteration

        policy, values, iters = policy_iteration(
            env=env,
            gamma=float(cfg.algo.gamma),
            theta=float(cfg.algo.theta),
            max_policy_eval_iters=int(cfg.algo.max_policy_eval_iters),
            max_policy_iters=int(cfg.algo.max_policy_iters),
        )
        return RunResult(policy=policy, values=values, iters=iters)
    if name == "mc_control":
        from .algos.mc.mc_control import mc_control_every_visit
        policy, Q, iters = mc_control_every_visit(
            env=env,
            gamma=float(cfg.algo.gamma),
            epsilon=float(cfg.algo.epsilon),
            episodes=int(cfg.algo.episodes),
            max_steps=int(cfg.algo.max_steps),
            seed=int(cfg.algo.seed),
        )
        # 用 max_a Q(s,a) 作为 V(s) 的近似展示
        values = Q.max(axis=1).tolist()
        return RunResult(policy=policy, values=values, iters=iters)
    if name == "sarsa":
        from .algos.td.sarsa import sarsa
        policy, Q, iters = sarsa(
            env=env,
            gamma=float(cfg.algo.gamma),
            alpha=float(cfg.algo.alpha),
            epsilon=float(cfg.algo.epsilon),
            episodes=int(cfg.algo.episodes),
            max_steps=int(cfg.algo.max_steps),
            seed=int(cfg.algo.seed),
        )
        values = Q.max(axis=1).tolist()
        return RunResult(policy=policy, values=values, iters=iters)
    if name == "q_learning":
        from .algos.td.q_learning import q_learning
        policy, Q, iters = q_learning(
            env=env,
            gamma=float(cfg.algo.gamma),
            alpha=float(cfg.algo.alpha),
            epsilon=float(cfg.algo.epsilon),
            episodes=int(cfg.algo.episodes),
            max_steps=int(cfg.algo.max_steps),
            seed=int(cfg.algo.seed),
        )
        values = Q.max(axis=1).tolist()
        return RunResult(policy=policy, values=values, iters=iters)
    if name == "dqn":
        from .algos.dqn.dqn import train_dqn

        res = train_dqn(
            env=env,
            gamma=float(cfg.algo.gamma),
            lr=float(cfg.algo.lr),
            batch_size=int(cfg.algo.batch_size),
            replay_capacity=int(cfg.algo.replay_capacity),
            start_learning_after=int(cfg.algo.start_learning_after),
            target_sync_interval=int(cfg.algo.target_sync_interval),
            total_steps=int(cfg.algo.total_steps),
            epsilon_start=float(cfg.algo.epsilon_start),
            epsilon_end=float(cfg.algo.epsilon_end),
            epsilon_decay_steps=int(cfg.algo.epsilon_decay_steps),
            double_dqn=bool(cfg.algo.double_dqn),
            dueling=bool(getattr(cfg.algo, "dueling", False)),
            n_step=int(getattr(cfg.algo, "n_step", 1)),
            prioritized_replay=bool(getattr(cfg.algo, "prioritized_replay", False)),
            prio_alpha=float(getattr(cfg.algo, "prio_alpha", 0.6)),
            prio_beta=float(getattr(cfg.algo, "prio_beta", 0.0)) or None,
            prio_beta_start=float(getattr(cfg.algo, "prio_beta_start", 0.4)),
            prio_beta_end=float(getattr(cfg.algo, "prio_beta_end", 1.0)),
            prio_eps=float(getattr(cfg.algo, "prio_eps", 1e-3)),
            noisy=bool(getattr(cfg.algo, "noisy", False)),
            seed=int(cfg.algo.seed),
        )
        # DQN 暂不计算 V(s)，只展示策略；用占位 0（连续观测时为空列表）
        if hasattr(env, "n_states"):
            values = [0.0 for _ in range(env.n_states)]
        else:
            values = []
        return RunResult(policy=res.policy, values=values, iters=res.iters)
    if name == "dyna_q":
        from .algos.model.dyna_q import dyna_q
        policy, Q, iters = dyna_q(
            env=env,
            gamma=float(cfg.algo.gamma),
            alpha=float(cfg.algo.alpha),
            epsilon=float(cfg.algo.epsilon),
            episodes=int(cfg.algo.episodes),
            max_steps=int(cfg.algo.max_steps),
            planning_steps=int(cfg.algo.planning_steps),
            seed=int(cfg.algo.seed),
        )
        values = Q.max(axis=1).tolist()
        return RunResult(policy=policy, values=values, iters=iters)
    if name == "mbpo_discrete":
        from .algos.model.mbpo_discrete import mbpo_discrete
        res = mbpo_discrete(
            env=env,
            gamma=float(cfg.algo.gamma),
            alpha=float(cfg.algo.alpha),
            epsilon=float(cfg.algo.epsilon),
            real_steps=int(cfg.algo.real_steps),
            model_train_episodes=int(cfg.algo.model_train_episodes),
            model_rollouts_per_update=int(cfg.algo.model_rollouts_per_update),
            model_horizon=int(cfg.algo.model_horizon),
            seed=int(cfg.algo.seed),
        )
        values = [0.0 for _ in range(env.n_states)]
        return RunResult(policy=res.policy, values=values, iters=res.iters)
    if name == "c51":
        from .algos.dqn.c51 import train_c51
        res = train_c51(
            env=env,
            gamma=float(cfg.algo.gamma),
            lr=float(cfg.algo.lr),
            batch_size=int(cfg.algo.batch_size),
            replay_capacity=int(cfg.algo.replay_capacity),
            total_steps=int(cfg.algo.total_steps),
            start_learning_after=int(cfg.algo.start_learning_after),
            target_sync_interval=int(cfg.algo.target_sync_interval),
            n_atoms=int(cfg.algo.n_atoms),
            v_min=float(cfg.algo.v_min),
            v_max=float(cfg.algo.v_max),
            seed=int(cfg.algo.seed),
        )
        values = [0.0 for _ in range(env.n_states)]
        return RunResult(policy=res.policy, values=values, iters=res.iters)
    if name == "reinforce":
        from .algos.pg.reinforce import train_reinforce
        res = train_reinforce(
            env=env,
            gamma=float(cfg.algo.gamma),
            lr=float(cfg.algo.lr),
            episodes=int(cfg.algo.episodes),
            max_steps=int(cfg.algo.max_steps),
            seed=int(cfg.algo.seed),
        )
        values = [0.0 for _ in range(env.n_states)]
        return RunResult(policy=res.policy, values=values, iters=res.iters)
    if name == "ppo":
        from .algos.pg.ppo import train_ppo
        res = train_ppo(
            env=env,
            gamma=float(cfg.algo.gamma),
            lam=float(cfg.algo.lam),
            lr=float(cfg.algo.lr),
            steps_per_epoch=int(cfg.algo.steps_per_epoch),
            epochs=int(cfg.algo.epochs),
            clip_ratio=float(cfg.algo.clip_ratio),
            train_iters=int(cfg.algo.train_iters),
            minibatch_size=int(getattr(cfg.algo, "minibatch_size", 0)) or None,
            vf_coef=float(cfg.algo.vf_coef),
            ent_coef=float(cfg.algo.ent_coef),
            target_kl=float(getattr(cfg.algo, "target_kl", 0.0)) or None,
            anneal_lr=bool(getattr(cfg.algo, "anneal_lr", False)),
            seed=int(cfg.algo.seed),
        )
        values = [0.0 for _ in range(env.n_states)]
        return RunResult(policy=res.policy, values=values, iters=res.iters)
    if name == "a2c":
        from .algos.pg.a2c import train_a2c
        res = train_a2c(
            env=env,
            gamma=float(cfg.algo.gamma),
            lr=float(cfg.algo.lr),
            steps_per_update=int(cfg.algo.steps_per_update),
            updates=int(cfg.algo.updates),
            vf_coef=float(cfg.algo.vf_coef),
            ent_coef=float(cfg.algo.ent_coef),
            seed=int(cfg.algo.seed),
        )
        values = [0.0 for _ in range(env.n_states)]
        return RunResult(policy=res.policy, values=values, iters=res.iters)
    if name == "td_lambda_eval":
        from .algos.td.td_lambda import td_lambda_prediction

        # 用“贪心向右”策略作为示例策略
        def pi(s: int) -> int:
            if s in env.terminal_states:
                return 0
            return 1

        values, iters = td_lambda_prediction(
            env=env,
            policy=pi,
            gamma=float(cfg.algo.gamma),
            lam=float(cfg.algo.lam),
            alpha=float(cfg.algo.alpha),
            episodes=int(cfg.algo.episodes),
            max_steps=int(cfg.algo.max_steps),
            seed=int(cfg.algo.seed),
        )
        # 从价值评估导出“向右贪心”的占位策略显示
        policy = [1 if s not in env.terminal_states else 0 for s in range(env.n_states)]
        return RunResult(policy=policy, values=values, iters=iters)
    if name == "td0_eval":
        from .algos.td.td0_eval import td0_prediction

        def pi(s: int) -> int:
            return 1 if s not in env.terminal_states else 0

        values, iters = td0_prediction(
            env=env,
            policy=pi,
            gamma=float(cfg.algo.gamma),
            alpha=float(cfg.algo.alpha),
            episodes=int(cfg.algo.episodes),
            max_steps=int(cfg.algo.max_steps),
            seed=int(cfg.algo.seed),
        )
        policy = [1 if s not in env.terminal_states else 0 for s in range(env.n_states)]
        return RunResult(policy=policy, values=values, iters=iters)
    if name == "sarsa_lambda":
        from .algos.td.sarsa_lambda import sarsa_lambda

        policy, Q, iters = sarsa_lambda(
            env=env,
            gamma=float(cfg.algo.gamma),
            lam=float(cfg.algo.lam),
            alpha=float(cfg.algo.alpha),
            epsilon=float(cfg.algo.epsilon),
            episodes=int(cfg.algo.episodes),
            max_steps=int(cfg.algo.max_steps),
            seed=int(cfg.algo.seed),
        )
        values = Q.max(axis=1).tolist()
        return RunResult(policy=policy, values=values, iters=iters)
    raise ValueError(f"未知算法: {name}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    console.rule("RL Overview — 训练入口")
    console.print("配置如下：")
    console.print(OmegaConf.to_yaml(cfg))

    env = _instantiate_env(cfg)
    result = _run_algo(cfg, env)

    console.rule("计算结果")
    console.print(f"迭代次数: {result.iters}")
    console.print(f"状态价值 V(s): {[round(v, 3) for v in result.values]}")
    # 策略打印为动作：0=左, 1=右
    console.print(f"最优策略 π*(s): {result.policy}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n中断退出。", style="yellow")
        sys.exit(1)
