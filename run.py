import numpy as np

from env.base import Env
from policy.linear import Policy
from evolve.train import NES
from evolve.visualize import plot


np.random.seed(0)
# params = np.random.randn(2)
params = np.array([0.001, 0.001])
params = NES('A', Env(), Policy, initial_params=params, num_episodes=10, lr=0.01, sigma=0.01, samples=10000)
print(params)
plot('A')
