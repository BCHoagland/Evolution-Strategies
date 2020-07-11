import numpy as np

from env.base import BaseEnv
from env.line import LineEnv

from policy.linear import Policy
from evolve.train import NES
from evolve.visualize import plot


np.random.seed(0)
params = np.array([-0.001, -0.001])
params = NES('A', BaseEnv(), Policy, initial_params=params, num_episodes=200, lr=1e-3, sigma=0.1, samples=10)
print(params)
plot('A')

np.random.seed(0)
params = np.array([-0.001, -0.001])
params = NES('B', LineEnv(), Policy, initial_params=params, num_episodes=200, lr=1e-9, sigma=0.1, samples=10)
print(params)
plot('B')
