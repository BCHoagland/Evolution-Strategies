import numpy as np
import torch

from env.base import BaseEnv
from env.line import LineEnv

from policy.linear import LinearPolicy
from policy.mlp import MLP

from evolve.train import NES
from evolve.visualize import plot


# np.random.seed(0)
# params = np.array([-0.001, -0.001])
# params = NES('A', BaseEnv(), LinearPolicy, initial_params=params, num_episodes=200, lr=1e-3, sigma=0.1, samples=10)
# print(params)
# plot('A')

# np.random.seed(0)
# params = np.array([-0.001, -0.001])
# params = NES('B', LineEnv(), LinearPolicy, initial_params=params, num_episodes=200, lr=1e-9, sigma=0.1, samples=10)
# print(params)
# plot('B')

np.random.seed(0)
torch.manual_seed(0)
params = np.random.randn(1153)
params = NES('MLP', BaseEnv(), MLP, initial_params=params, num_episodes=200, lr=3e-4, sigma=0.1, samples=10)
print(params)
plot('MLP')
