# RL Overview (代码即教材)

本项目以“代码即教材”为目标，配合 Hydra 与 uv 管理，内容覆盖：值函数、策略梯度、模型驱动、多智能体、以及 LLM×RL 等专题。

## 安装

```bash
uv sync                 # 安装基础依赖（numpy, hydra-core, rich, tqdm）
uv sync --extra torch   # 可选：安装 PyTorch
uv sync --extra gym     # 可选：安装 Gymnasium
```

## 运行示例

```bash
uv run -m rl_overview.train
uv run -m rl_overview.train env.n_states=9
# 算法切换（值/TD/MC/模型/深度/PG）
uv run -m rl_overview.train algo=policy_iteration
uv run -m rl_overview.train algo=q_learning algo.episodes=200
uv run -m rl_overview.train algo=dyna_q

# 规划演示（模型已知）
uv run -m rl_overview.cli.plan_run algo=mcts

# Bandit
uv run -m rl_overview.cli.bandits_run env=bernoulli_bandit algo=ucb1
```

## 可选组件
- Torch 工具/网络：位于 `src/rl_overview/nn/`，默认不导入，未安装 torch 也可运行。
- Gym 包装器：`rl_overview.envs.wrappers` 提供 `gym_to_step_env` 将 Gymnasium 环境适配到最小 StepEnv 接口。

## 深度 RL（需安装 torch）

```bash
uv sync --extra torch
# DQN（支持：Double、Dueling、n-step、Prioritized Replay、Noisy 可选）
uv run -m rl_overview.train algo=dqn algo.total_steps=2000 algo.start_learning_after=200

# REINFORCE / A2C / PPO（策略梯度）
uv run -m rl_overview.train algo=reinforce algo.episodes=100
uv run -m rl_overview.train algo=a2c updates=5
uv run -m rl_overview.train algo=ppo steps_per_epoch=512 epochs=2

# 在 Gym（CartPole）上运行（需 --extra gym）
uv sync --extra gym
uv run -m rl_overview.cli.gym_run env=gym_cartpole algo=dqn algo.total_steps=5000
uv run -m rl_overview.cli.gym_run env=gym_cartpole algo=ppo steps_per_epoch=1024 epochs=5
```

## 多智能体（MARL）

```bash
uv run -m rl_overview.cli.marl_run env=normal_form_matching_pennies
```
