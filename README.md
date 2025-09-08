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
```

## 可选组件
- Torch 工具/网络：位于 `src/rl_overview/nn/`，默认不导入，未安装 torch 也可运行。
- Gym 包装器：`rl_overview.envs.wrappers` 提供 `gym_to_step_env` 将 Gymnasium 环境适配到最小 StepEnv 接口。

