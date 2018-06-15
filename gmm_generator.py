'''
Usage: gmm-generator.py PATH [options]

Arguments:
    PATH    path to example file

Options:
    -l, --limit=<int>           Limit on the number of parsed lines
    -n, --n-components=<int>    Number of gaussians for each mixture
                                [default: 5]
    --log-level=<int>           Log level :)
                                [default: 0]
    --scale-data                Boolean flag if one should scale data
'''
import numpy as np

from utils.data import get_mnist, StandardScaller
from utils.plotting import plot_sample, plot_class_samples

from typeopt import Arguments
from sklearn.mixture import BayesianGaussianMixture


class GmmGenerator(object):

    def __init__(self, n_components=10, verbose=0):
        self.params = {}
        self.verbose = verbose
        self.n_components = n_components

    def fit(self, X, Y):
        ''' fit the gaussians '''
        self.num_classes = len(set(Y))
        self.prob = np.zeros(self.num_classes)
        for y in range(self.num_classes):
            X_y = X[np.where(Y == y)]
            gmm = BayesianGaussianMixture(
                self.n_components,
                verbose=self.verbose
            )

            self.params[y] = gmm.fit(X_y)
            self.prob[y] = len(X_y) / len(X)

        return self

    def sample(self, shape=(28, 28)):
        ''' Samples from trained distributions'''
        y = np.random.choice(self.num_classes, p=self.prob)
        return self.sample_from_class(y, shape)

    def sample_from_class(self, y, shape=(28, 28)):
        sample, z = self.params[y].sample()
        mean = self.params[y].means_[z]

        if shape:
            sample = np.reshape(sample, shape)
            mean = np.reshape(mean, shape)

        return sample, mean


if __name__ == '__main__':
    args = Arguments(__doc__)
    print(args)
    print("Loading data from %s" % args.path)
    X, Y = get_mnist(args.path, limit=args.limit)
    print("Loaded %i samples" % len(X))

    if args.scale_data:
        print("Normalize data")
        X = StandardScaller().fit(X).transform(X)

    print('Training GmmGenerator')
    generator = GmmGenerator(args.n_components, args.log_level)
    generator = generator.fit(X, Y)

    # Plotting
    plot_sample(generator)
    plot_class_samples(generator)
