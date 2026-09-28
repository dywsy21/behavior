"""Full-path Flow-Noise PPO primitives (all likelihood arithmetic is fp32).

An environment action is the ENTIRE 10 x 32 x 23 latent transition path.
Only its decoded first 16 controls execute. The remaining correlated positions
must still be scored. The initial standard normal is parameter independent.
"""
from __future__ import annotations

import math
import torch
from torch import nn

PAD_DIMS = (7, 8, 17, 18)
VALID_DIMS = tuple(i for i in range(27) if i not in PAD_DIMS)


def gaussian_logp(value, mean, std):
    """One scalar per path; accepts [...,32,27], never averages dimensions."""
    x, m, s = (v.float()[..., VALID_DIMS] for v in (value, mean, std))
    if not torch.isfinite(s).all() or not (s > 0).all():
        raise ValueError('Path likelihood requires strictly positive finite std')
    return (-.5 * ((x-m)/s).square() - s.log() - .5*math.log(2*math.pi)).sum()


def gaussian_kl(old_mean, old_std, mean, std):
    om, os, m, s = (v.float()[..., VALID_DIMS] for v in (old_mean, old_std, mean, std))
    return ((s/os).log() + (os.square() + (om-m).square())/(2*s.square()) - .5).sum()


def ppo_objective(logp, old_logp, advantage, clip=.05):
    delta = logp.float() - old_logp.float()
    # Overflow is a failed update, not an implicit modification of PPO.
    if not torch.isfinite(delta).all() or (delta.abs() > 60).any():
        raise FloatingPointError('Unsafe full-path log-ratio')
    ratio = delta.exp()
    return -torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage).mean()


class NoiseHead(nn.Module):
    """Conditioned on frozen public features, current latent and diffusion time."""
    def __init__(self, feature_dim=2048, low=.003, high=.03, initial=.01):
        super().__init__()
        if not 0 < low < initial < high:
            raise ValueError('Invalid noise interval')
        self.low, self.high = low, high
        self.feature = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 32), nn.Tanh())
        self.net = nn.Sequential(nn.Linear(32+27+1, 64), nn.Tanh(), nn.Linear(64, 27))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log((initial-low)/(high-initial)))

    def forward(self, feature, latent, time):
        f = self.feature(feature.detach().float()).unsqueeze(1).expand(-1, latent.size(1), -1)
        t = time.float()[:, None, None].expand(-1, latent.size(1), 1)
        z = self.net(torch.cat([f, latent.detach().float(), t], -1))
        return self.low + (self.high-self.low)*z.sigmoid()


class ValueHead(nn.Module):
    """No privileged actor channel; first critic uses public frozen features."""
    def __init__(self, feature_dim=2048):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 256),
                                 nn.Tanh(), nn.Linear(256, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature):
        return self.net(feature.detach().float()).squeeze(-1)


def chunk_reward(rewards, gamma=.9998):
    return sum(gamma**i * float(r) for i, r in enumerate(rewards))


def gae(transitions, gamma=.9998, lam=.95):
    """Single worker, chronological. Bootstrap truncation, never cross reset.

    Each entry contains value, next_value (FINAL observation before reset),
    rewards, terminated, truncated, episode. Collector boundaries bootstrap.
    """
    advantages = [0.] * len(transitions)
    carry = 0.
    for i in range(len(transitions)-1, -1, -1):
        row = transitions[i]
        m = len(row['rewards'])
        if m == 0:
            raise ValueError('A PPO transition must execute at least one control')
        discount = gamma ** m
        bootstrap = 0. if row['terminated'] else float(row['next_value'])
        delta = chunk_reward(row['rewards'], gamma) + discount*bootstrap - float(row['value'])
        continues = (not row['terminated'] and not row['truncated'] and i+1 < len(transitions)
                     and transitions[i+1]['episode'] == row['episode'])
        carry = delta + (discount*lam*carry if continues else 0.)
        advantages[i] = carry
    returns = [a+float(t['value']) for a,t in zip(advantages,transitions)]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


class ControlBudget:
    """Reserve BEFORE dispatch, including expert and benchmark actions."""
    def __init__(self, limit=10000):
        self.limit, self.used, self.pending = int(limit), 0, 0

    @property
    def remaining(self):
        return self.limit-self.used-self.pending

    def reserve(self, count):
        if type(count) is not int or count < 1 or count > self.remaining:
            raise RuntimeError('Actual-control budget exhausted or invalid reservation')
        self.pending += count

    def finish(self, reserved, actual):
        if not 0 <= actual <= reserved <= self.pending:
            raise ValueError('Invalid actual-control accounting')
        self.pending -= reserved
        self.used += actual
