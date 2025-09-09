from __future__ import annotations

"""
偏好/奖励模型常用损失。
"""

import numpy as np


def pairwise_logistic_loss(score_pos, score_neg):
    """
    Bradley-Terry / Logistic 偏好损失：
    L = -log( sigmoid(score_pos - score_neg) )
    支持 numpy 或 torch 张量（若传 torch 则返回同设备张量）。
    """
    try:
        import torch
        if isinstance(score_pos, torch.Tensor):
            return torch.nn.functional.softplus(- (score_pos - score_neg)).mean()
    except Exception:
        pass
    x = np.asarray(score_pos) - np.asarray(score_neg)
    return float(np.mean(np.log1p(np.exp(-x))))

