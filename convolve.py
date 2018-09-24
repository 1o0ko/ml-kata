import numpy as np
import matplotlib.pyplot as plt


def convolve(X, W, b, p, s):
    ''' Naive implementation of convolution operator
    Arguments:
        X - image to be convolved
        W - image kernel
        b - bias term

       p - padding
       s - stride
    '''

    # width, height, channels
    n_w, n_h, c = X.shape

    # get kernal size
    k, k, c = W.shape

    n_w_c = (n_w - k + 2 * p) // s + 1
    n_h_c = (n_h - k + 2 * p) // s + 1

    # create oupt matrix
    X_conv = np.zeros(shape=(n_w_c, n_h_c, c))

    # Padd image (this is faster than np.pad!)
    X_pad = np.zeros((n_w + 2*p, n_h + 2*p, c))
    X_pad[p:+ n_w, p:n_h + p, :] = X

    # convolve
    for i in range(0, n_w_c):
        for j in range(0, n_h_c):
            X_slice = X_pad[s * i:(k + s * i), s * j:(k + s * j), :]
            assert X_slice.shape == (k, k, c)
            X_conv[i, j] = np.sum(np.multiply(X_slice, W)) + float(b)

    return X_conv, X_pad


def main():
    X = np.random.randn(10, 10, 3)

    # width, height, channels
    n_w, n_h, c = X.shape

    # kernel size, stride, padding
    k, s, p = 4, 1, 2

    W = np.random.randn(k, k, c)
    b = np.random.randn(1, 1, 1)

    X_conv, X_pad = convolve(X, W, b, s, p)

    # Plotting
    images = [
        (X, 'image'),
        (X_pad, 'padded image'),
        (W, 'kernel'),
        (X_conv, 'convoled image')]

    f, axarr = plt.subplots(1, len(images), figsize=(20, 10))
    for ax, (image, title) in zip(axarr, images):
        ax.imshow(image)
        ax.title.set_text(title)


if __name__ == '__main__':
    import sys
    sys.exit(main())
