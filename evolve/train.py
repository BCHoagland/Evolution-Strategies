import numpy as np

from evolve.utils import save


def fitness(env, policy_class, params):
    '''
    Calculate the fitness score for the given policy parameters
    '''
    policy = policy_class(params)

    cum_r = 0
    s = env.reset()
    while True:
        # interact with environment
        a = policy(s)
        s2, r, done = env.step(a)
        cum_r += r

        # either move to next step or end the episode
        if done:
            return cum_r
        s = s2


#! TODO: generalize this so that the plotting/bookkeeping is separate from the NES/param updates
def NES(name, env, policy_class, initial_params, num_episodes, lr, sigma, samples):
    '''
    Run optimization using the NES algorithm
    '''
    fitness_history = []

    params = initial_params

    for _ in range(num_episodes):

        # estimate gradient
        grad = 0
        for _ in range(samples):
            eps = np.random.randn(*params.shape)
            grad += fitness(env, policy_class, params + sigma * eps) * eps
        grad /= samples * sigma

        # update parameters
        params += lr * grad

        # record the fitness of current params
        fitness_history.append(fitness(env, policy_class, params))

    # save fitness history
    save(fitness_history, name)

    return params
