import numpy as np
from cmaes import CMA


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

if __name__ == "__main__":
    cma_es = CMA(mean=np.zeros(2), sigma=1.3)

    for generation in range(20):
        solutions = []
        for _ in range(cma_es.population_size):
            x = cma_es.ask()
            value = quadratic(*x)
            solutions.append((x, value))
            print(f"#{generation} {round(value, 2)}  \t(x1={round(x[0], 2)}, x2 = {round(x[1], 2)})")
        cma_es.tell(solutions)