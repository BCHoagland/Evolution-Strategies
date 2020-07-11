import pickle


def save(obj, filename):
    with open(f'data/saved/{filename}', 'wb') as f:
        pickle.dump(obj, f)
