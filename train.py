import numpy as np

from env import Env
# from model import Policy
from policy import Policy


α = 0.01
σ = 0.01
n = 10000


env = Env()


def F(θ):
    '''
    Calculate the fitness score for the given policy parameters
    '''
    π = Policy(θ)

    cum_r = 0
    s = env.reset()
    while True:
        # interact with environment
        a = π(s)
        s2, r, done = env.step(a)
        cum_r += r

        # either move to next step or end the episode
        if done:
            return cum_r
        s = s2


def train(num_episodes):
    # set initial params
    θ = np.random.randn(2)

    for _ in range(num_episodes):

        # estimate gradient
        grad = 0
        for _ in range(n):
            ε = np.random.randn()
            grad += F(θ + σ * ε) * ε
        grad /= n * σ

        # update parameters
        θ += α * grad

    return θ


θ = train(100)
print(θ)
