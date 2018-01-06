'''
Sketch of a fowrard pass of model described in Character Aware Language Models
(https://arxiv.org/abs/1508.06615)
'''
import numpy as np


def word_to_indices(word, c2i):
    return [c2i[char] for char in word if char in c2i]


def tokenize(sentence, c2i):
    tokens = sentence.split(" ")
    for token in tokens:
        idxs = word_to_indices(token, c2i)
        # can return an empty list
        if idxs:
            yield idxs


def get_word_matrix(idxs, lookup_table):
    return np.hstack([lookup_table[idx].reshape(-1, 1) for idx in idxs])


def correct_padding(n, k, p):
    ''' If filter is longer than our matrix we have to padd the matrix'''
    d = k - n
    if d > 0:
        return d

    return p


def convolve(X, W, s=1, p_w=0, p_h=0):
    '''
    General function to convolve matrix with non-symetric kernel
    '''
    # width, height, channels
    n_w, n_h = X.shape
    k_w, k_h = W.shape

    p_w = correct_padding(n_w, k_w, p_w)
    p_h = correct_padding(n_h, k_h, p_h)

    n_w_c = (n_w - k_w + 2 * p_w) // s + 1
    n_h_c = (n_h - k_h + 2 * p_h) // s + 1

    # create oupt matrix
    X_conv = np.zeros(shape=(n_w_c, n_h_c))

    # padd source image
    X_pad = np.pad(X, [(p_w, p_w), (p_h, p_h)], 'constant', constant_values=0)

    # convolve
    for i in range(0, n_w_c):
        for j in range(0, n_h_c):
            X_slice = X_pad[s * i:(k_w + s * i), s * j:(k_h + s * j)]
            assert X_slice.shape == (k_w, k_h)
            X_conv[i, j] = np.sum(np.multiply(X_slice, W))

    return X_conv

# Model sketch


def relu(x):
    return x * (x > 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Affine(object):
    def __init__(self, input_size, output_size):
        self.W = np.random.random((output_size, input_size))
        self.b = np.random.random((output_size, 1))

    def __call__(self, x):
        return self.W.dot(x) + self.b


class HighwayLayer(object):
    def __init__(self, input_size, g=relu):
        self.A_h = Affine(input_size, input_size)
        self.A_t = Affine(input_size, input_size)
        self.g = g

    def __call__(self, y):
        # calulate transform gate
        t = sigmoid(self.A_t(y))

        # calculate output layer
        return t * self.g(self.A_h(y)) + (1 - t) * y

# Model hyperparameters


# char embedding size
d = 15

# target word embedding size
D = 100

# sample char vocabulary
c2i = {chr(i): index for index, i in enumerate(range(ord('a'), ord('z') + 1))}

# sample char embeddings
char_embeddings = 0.001 * np.random.random((len(c2i), d))

# filter sizes
w_sizes = [1, 2, 3, 4, 5, 6]

# how many times should we replicate the filters to obtain the word
# embedding of approriate size
w_n = 100 // len(w_sizes)

# Create filters
filters = [np.random.random((d, w))
           for w_size in w_sizes
           for w in w_n * [w_size]]

if len(filters) < D:
    filters.extend([np.random.random((d, w))
                    for w in random.choices(w_sizes, k=D - len(filters))])

sentence = "I think this is stupid , but necessary"

highway = HighwayLayer(D)

for word in tokenize(sentence, c2i):
    C = get_word_matrix(word, char_embeddings)

    # apply 'max-pooling over time' to get one float for filter.
    W = np.array([np.max(np.tanh(convolve(C, H) + 0.01))
                  for H in filters]).reshape(-1, 1)
    W = highway(W)

    print("---")
    print("C: %s, %s" % C.shape)
    print("W: %s, %s" % W.shape)
