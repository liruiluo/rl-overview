from __future__ import annotations

from typing import Optional


def is_torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def get_device(prefer: Optional[str] = None):
    """
    返回 torch 设备（如可用）。
    - 未安装 torch 时，返回字符串 'cpu' 以避免硬依赖。
    - prefer 可指定 'cuda'/'mps'/'cpu'，在可用时优先选择。
    """
    if not is_torch_available():
        return "cpu"

    import torch

    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")

    # 自动选择：优先 CUDA，再 MPS，最后 CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

