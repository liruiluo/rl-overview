from __future__ import annotations

import math
import torch
import torch.nn as nn


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer("eps_weight", torch.empty(out_features, in_features))
        self.register_buffer("eps_bias", torch.empty(out_features))
        self.reset_parameters(sigma0)

    def reset_parameters(self, sigma0: float):
        mu_range = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.mu_weight, -mu_range, mu_range)
        nn.init.uniform_(self.mu_bias, -mu_range, mu_range)
        nn.init.constant_(self.sigma_weight, sigma0 / math.sqrt(self.in_features))
        nn.init.constant_(self.sigma_bias, sigma0 / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        eps_in = self._scale_noise(self.in_features).to(x.device)
        eps_out = self._scale_noise(self.out_features).to(x.device)
        self.eps_weight = eps_out.ger(eps_in)
        self.eps_bias = eps_out
        w = self.mu_weight + self.sigma_weight * self.eps_weight
        b = self.mu_bias + self.sigma_bias * self.eps_bias
        return torch.nn.functional.linear(x, w, b)

