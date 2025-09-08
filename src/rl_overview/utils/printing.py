from __future__ import annotations

from typing import Iterable


def arrows_for_policy(policy: Iterable[int]) -> str:
    """
    将 0/1 动作策略渲染为 ←/→ 便于肉眼检查。
    """
    mapping = {0: "←", 1: "→"}
    return "".join(mapping.get(int(a), "?") for a in policy)

