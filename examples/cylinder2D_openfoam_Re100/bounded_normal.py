# Adapted from https://github.com/tensorforce/tensorforce/blob/master/tensorforce/core/distributions/gaussian.py

import torch
from torch.distributions import Normal
from torch.distributions.utils import _standard_normal

import math

EPSILON = 1e-6
LOG_EPSILON = math.log(1e-6)


def adjust_stddev(stddev):
    # Clip stddev for numerical stability (epsilon < 1.0, hence negative)
    stddev = torch.clip(stddev, LOG_EPSILON, -LOG_EPSILON)
    # Softplus transformation (based on https://arxiv.org/abs/2007.06059)
    stddev = 0.25 * (torch.log(1.0 + torch.exp(stddev)) + 0.2) / (math.log(2.0) + 0.2)

    return stddev


class BoundedNormal(Normal):
    r"""
    Gaussian distribution, for bounded-continuous actions.

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    def __init__(self, loc, scale, bounded_transform='tanh', action_spec=None, validate_args=None):
        scale = adjust_stddev(scale)
        super(BoundedNormal, self).__init__(loc, scale, validate_args)

        self.log_scale = torch.log(self.scale + EPSILON)

        self.action_spec = action_spec
        self.action_min_value = None
        self.action_max_value = None

        if action_spec is not None:
            self.action_max_value = torch.Tensor(action_spec.high)
            self.action_min_value = torch.Tensor(action_spec.low)

            self.action_scale = (self.action_max_value - self.action_min_value) / 2.0
            self.action_bias = (self.action_max_value + self.action_min_value) / 2.0

        if bounded_transform is None:
            bounded_transform = 'tanh'

        if bounded_transform not in ('clipping', 'tanh'):
            raise ValueError(f'{bounded_transform } is not supported!')
        elif bounded_transform == 'tanh' and (
            (self.action_min_value is not None) is not (self.action_max_value is not None)
        ):
            raise ValueError(f'{bounded_transform} does not support one-sided bounded action space!')
        elif self.action_min_value is None and self.action_max_value is None:
            raise ValueError('Use the built-in Normal distribution for a non-bounded action space!')

        self.bounded_transform = bounded_transform

    def sample(self, temperature, sample_shape=torch.Size()):
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            action = None

            if temperature < EPSILON:
                action = self.loc.expand(shape)
            else:
                eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
                action = self.loc + eps * self.scale * temperature

            # Bounded transformation
            if self.bounded_transform == 'tanh':
                action = torch.tanh(action)
            elif self.bounded_transform == 'clipping':
                action = torch.clip(action, -1.0, 1.0)

            if self.action_min_value is not None and \
                    self.action_max_value is not None:
                action = action * self.action_scale + self.action_bias

        return action

    def log_prob(self, action):
        if self._validate_args:
            self._validate_sample(action)
        # compute the variance
        var = (self.scale ** 2) + EPSILON

        # Inverse bounded transformation
        if self.action_min_value is not None and self.action_max_value is not None:
            action = 2.0 * (action - self.action_min_value) / (self.action_max_value - self.action_min_value) - 1.0

        if self.bounded_transform == 'tanh':
            clip = 1.0 - EPSILON
            action = torch.clip(action, -clip, clip)
            action = torch.atanh(action)

        log_prob = -((action - self.loc) ** 2) / (2 * var) - self.log_scale - math.log(math.sqrt(2 * math.pi))

        if self.bounded_transform == 'tanh':
            log_prob -= 2.0 * (math.log(2.0) - action - torch.log(1.0 + torch.exp(-2.0 * action)))

        return log_prob

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale + EPSILON)
