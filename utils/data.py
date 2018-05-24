import numpy as np
import pandas as pd


class StandardScaller(object):

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X, back=False):
        if back:
            return self.std * X + self.mean
        else:
            (X - self.mean) / self.std


class Counter():
    def __init__(self, i=0):
        self.i = i

    def value(self):
        self.i += 1
        return self.i


def get_mnist(path, shuffle=True, limit=None):
    '''
    loads mnist data from kaggle competition

    Arguments:
        path - path to the mnist train.csv data
    '''
    data = pd.read_csv(path, nrows=limit).as_matrix()
    if shuffle:
        np.random.shuffle(data)

    X = data[:, 1:] / 255.0  # data is from 0..255
    Y = data[:, 0]

    return X, Y
