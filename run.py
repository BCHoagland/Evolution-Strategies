from policy.mlp import MLP

from evolve.train import NES
from evolve.visualize import plot


params = NES('MLP', 'CartPole-v1', MLP, n_in=4, n_h=32, n_out=2, num_episodes=100, lr=3e-4, sigma=0.1, samples=10, seed=0)
plot('MLP')
