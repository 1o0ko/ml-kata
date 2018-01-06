import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = np.random.randn(10, 10, 3)

    # width, height, channels
    n_w, n_h, c = X.shape

    # kernel size, stride, padding
    k, s, p = 4, 1, 2

    W = np.random.randn(k, k, c)
    b = np.random.randn(1, 1, 1)

    n_w_c = (n_w - k + 2 * p)//s + 1
    n_h_c = (n_h - k + 2 * p)//s + 1

    # create oupt matrix
    X_conv = np.zeros(shape=(n_w_c, n_h_c, c))

    # padd source image
    X_pad = np.pad(X, [(p, p),(p, p),(0,0)], 'constant',  constant_values = 0)

    # convolve
    for i in range(0, n_w_c):
        for j in range(0, n_h_c):
            X_slice = X_pad[s*i:(k +s*i),s*j:(k+s*j),:]
            assert X_slice.shape == (k, k, 3)
            X_conv[i,j] = np.sum(np.multiply(X_slice, W)) + float(b)


    # Plotting 
    images = [
        (X,      'image'), 
        (X_pad,  'padded image'),  
        (W,      'kernel'),
        (X_conv, 'convoled image')]

    f, axarr = plt.subplots(1, len(images), figsize=(20, 10))
    for ax, (image, title) in zip(axarr, images):
        ax.imshow(image)
        ax.title.set_text(title)
