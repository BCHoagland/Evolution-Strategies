import numpy as np
import torch
import gym

from evolve.utils import save


def fitness(env, policy, params):
    '''
    Calculate the fitness score for the given policy parameters
    '''
    policy.set_params(params)

    cum_r = 0
    s = env.reset()
    while True:
        # interact with environment
        with torch.no_grad():
            a = policy(s).numpy()
        s2, r, done, _ = env.step(a)
        cum_r += r

        # either move to next step or end the episode
        if done:
            return cum_r
        s = s2


def NES(name, env_name, policy_class, n_in, n_h, n_out, num_episodes, lr, sigma, samples, seed=0):
    '''
    Run optimization using the NES algorithm
    '''
    # create env
    env = gym.make(env_name)

    # set random seeds
    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # dummy policy whose parameters we'll overwrite a lot
    policy = policy_class(n_in, n_h, n_out)

    # get the initial network params
    params = np.random.randn(policy.num_params())
    # params = np.random.randn(num_params(n_in, n_h, n_out))

    # where we'll store the progress of our agent
    fitness_history = []

    # train for a set number of episodes
    for _ in range(num_episodes):

        # estimate gradient
        grad = 0
        for _ in range(samples // 2):
            # add to the gradient estimate (with mirror sampling)
            eps = np.random.randn(*params.shape)
            grad += fitness(env, policy, params + sigma * eps) * eps
            grad += fitness(env, policy, params - sigma * eps) * (-eps)

        # finish the gradient estimate
        grad /= samples * sigma

        # update parameters
        params += lr * grad

        # record the fitness of current params
        fitness_history.append(fitness(env, policy, params))

    # save fitness history
    save(fitness_history, name)

    return params
