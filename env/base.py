import numpy as np

class Env:
    def __init__(self):
        self.fn = lambda x: 3 * x

    def reset(self):
        self.s = np.random.rand() * 10 - 5
        return self.s

    def step(self, a):
        error = (self.fn(self.s) - a) ** 2
        s = self.reset()
        return s, -error, True
