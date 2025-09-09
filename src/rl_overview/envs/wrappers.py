from __future__ import annotations

from typing import Any, Dict, Tuple


class GymStepEnv:
    """
    将 Gymnasium 环境适配为最小 StepEnv 接口：
    - reset(seed) -> obs
    - step(a) -> (obs, reward, terminated, info)
    注意：Gymnasium 的 (terminated, truncated) 合并为一个 terminated。
    """

    def __init__(self, env: Any):
        self._env = env
        # 暴露通用属性，便于算法侧自适应
        try:
            self.n_actions = int(env.action_space.n)
        except Exception:
            self.n_actions = None  # 连续动作场景暂不支持
        try:
            shape = tuple(env.observation_space.shape)
        except Exception:
            shape = None
        self.obs_shape = shape

    def reset(self, seed: int | None = None):
        if seed is not None:
            try:
                obs, info = self._env.reset(seed=seed)
            except TypeError:
                # 兼容旧 API
                self._env.reset(seed)
                obs, info = self._env.reset()
        else:
            obs, info = self._env.reset()
        return obs

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        out = self._env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return obs, float(reward), bool(terminated or truncated), info
        # 兼容旧 Gym
        obs, reward, done, info = out
        return obs, float(reward), bool(done), info


def gym_to_step_env(env_id: str, **kwargs) -> GymStepEnv:
    """
    工厂方法：按需导入 gymnasium 并创建包装环境。
    使用示例：
        step_env = gym_to_step_env("CartPole-v1")
        obs = step_env.reset(42)
        obs, r, done, info = step_env.step(env.action_space.sample())
    """
    try:
        import gymnasium as gym
    except Exception as e:  # pragma: no cover - 仅在未安装时触发
        raise RuntimeError(
            "未安装 gymnasium，可用 'uv add --extra gym .' 或 'uv sync --extra gym' 安装"
        ) from e

    env = gym.make(env_id, **kwargs)
    return GymStepEnv(env)
