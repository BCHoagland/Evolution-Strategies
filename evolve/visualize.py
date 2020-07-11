import pickle
import matplotlib.pyplot as plt


def plot(filename, show=False):
    with open(f'data/saved/{filename}', 'rb') as f:
        # load the data to plot
        data = pickle.load(f)

        # format and plot
        plt.title(filename)
        plt.xlabel('Parameter Updates')
        plt.ylabel('Fitness')
        plt.plot(range(1, len(data) + 1), data)

        # save the figure
        plt.savefig(f'data/img/{filename}')

        # show the plot if necessary
        if show:
            plt.show()
