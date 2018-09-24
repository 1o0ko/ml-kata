'''
Numpy implementation of a vectorized convolution operation

Based on: https://sgugger.github.io/convolution-in-depth.html#convolution-in-depth
'''
import numpy as np


def vectorize(X, kernel_size, stride=1, padding=0):
    '''
    Use np.take and broadcasting to speed up convolution
    '''
    k1, k2 = kernel_size
    n_rows, n_cols = X.shape

    n_rows_padded, n_cols_padded = n_rows + 2 * padding, n_cols + 2 * padding

    # initialize with zeros
    X_padded = np.zeros((n_rows_padded, n_cols_padded))

    # add padding
    X_padded[padding:n_rows + padding, padding:n_cols + padding] = X

    # Build an array of indices that map to our kernel
    grid = np.array([j + n_cols_padded * i
                     for i in range(k1)
                     for j in range(k2)])

    # Build an array of indices that correspond to starting point of
    # each top-left corner of the kernel position
    start_idx = np.array([j + n_cols_padded * i
                          for i in range(0, n_rows_padded - k1 + 1, stride)
                          for j in range(0, n_cols_padded - k2 + 1, stride)])

    # use broacasting to add kernel indices to each kernel starting point
    # (N, 1) + (1, M) ~ (N, M)
    to_take = start_idx.reshape(-1, 1) + grid.reshape(1, -1)

    # Use np.take to create view on X using our calculated indices
    return X_padded.take(to_take)


class CNN:
    def __init__(self, weights, stride=1, padding=0):
        ''' Weights could be initialized here '''
        self.weights = weights
        self.kernel_size = weights.shape
        self.biases = np.zeros(self.kernel_size[-1])

        self.stride = stride
        self.padding = padding

    def forward(self, X):
        # get shapes
        n_rows, n_cols = X.shape
        k_h, k_w, n_c = self.kernel_size

        X_vec = vectorize(X, (k_h, k_w), self.stride, self.padding)

        # it's a feedforward neural network!
        y = X_vec @ self.weights.reshape(k_h * k_w, -1) + self.biases

        # reshape back to it's size
        n_rows = (n_rows - k_h + 2 * self.padding) // self.stride + 1
        n_cols = (n_cols - k_w + 2 * self.padding) // self.stride + 1

        return y.reshape(n_rows, n_cols, n_c)

    def backward(self, grad):
        pass

    def __call__(self, X):
        return self.forward(X)
