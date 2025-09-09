from __future__ import annotations

"""
Direct Preference Optimization (DPO) 损失实现（不含模型前向）：
给定策略与参考模型对“被偏好/被拒绝”响应的 log-prob，计算 DPO 损失。
"""

import numpy as np


def dpo_loss(policy_logp_pos, policy_logp_neg, ref_logp_pos, ref_logp_neg, beta: float = 0.1):
    """
    L = - E[ log sigma( beta * ( (logπ_pos - logπ_neg) - (logπ_ref_pos - logπ_ref_neg) ) ) ]
    可输入 numpy 或 torch；若是 torch 则返回 torch 张量。
    """
    try:
        import torch
        if isinstance(policy_logp_pos, torch.Tensor):
            z = beta * ((policy_logp_pos - policy_logp_neg) - (ref_logp_pos - ref_logp_neg))
            return torch.nn.functional.softplus(-z).mean()
    except Exception:
        pass
    ppos, pneg = np.asarray(policy_logp_pos), np.asarray(policy_logp_neg)
    rpos, rneg = np.asarray(ref_logp_pos), np.asarray(ref_logp_neg)
    z = beta * ((ppos - pneg) - (rpos - rneg))
    return float(np.mean(np.log1p(np.exp(-z))))

