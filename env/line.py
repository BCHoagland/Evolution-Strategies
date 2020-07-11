import numpy as np

class LineEnv:
    def __init__(self):
        self.max_t = 10

    def reset(self):
        self.t = 0

        self.s = np.random.rand() * 10 - 5
        return self.s

    def step(self, a):
        cost = self.s**2 + a**2
        self.s += self.s + a

        self.t += 1
        # done = True if self.t > self.max_t or abs(self.s) < 0.01 else False
        done = True if self.t > self.max_t else False

        return self.s, -cost, done