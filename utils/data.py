import numpy as np
import pandas as pd


def get_mnist(path, limit=None):
    '''
    loads mnist data from kaggle competition

    Arguments:
        path - path to the mnist train.csv data
    '''
    print("Reading in and transforming data...")
    df = pd.read_csv(path)
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0  # data is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y
