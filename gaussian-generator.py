'''
Usage: gaussian-generator.py PATH [options]

Arguments:
    PATH    path to example file

Options:
    -l, --limit=<int>           Limit on the number of parsed lines
    --scale-data                Boolean flag if one should scale data
'''
import numpy as np

from utils.data import get_mnist
from utils.plotting import plot_sample, plot_class_samples
from typeopt import Arguments


class StandardScaller(object):

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean 


class GaussianGenerator(object):

    def __init__(self):
        self.params = {}

    def fit(self, X, Y):
        ''' fit the gaussians '''
        self.num_classes = len(set(Y))
        self.prob = np.zeros(self.num_classes)
        for y in range(self.num_classes):
            X_y = X[np.where(Y == y)]
            self.params[y] = {
                "mean": np.mean(X_y, axis=0),
                "covarience": np.cov(X_y.T)
            }

            self.prob[y] = len(X_y) / len(X)

        return self

    def predict(self, X):
        '''
        Predicts the most probable class
        '''
        pass

    def sample(self, y=None, shape=(28, 28)):
        '''
        Samples from trained distributions
        '''
        if not y:
            y = np.random.choice(self.num_classes, p=self.prob)

        sample = np.random.multivariate_normal(
            mean=self.params[y]['mean'],
            cov=self.params[y]['covarience'])

        if shape:
            sample = np.reshape(sample, shape)

        return sample, y


if __name__ == '__main__':
    args = Arguments(__doc__)
    print(args)
    print("Loading data from %s" % args.path)
    X, Y = get_mnist(args.path, limit=args.limit)
    print("Loaded %i samples" % len(X))

    if args.scale_data:
        print("Normalize data")
        X = StandardScaller().fit(X).transform(X)

    print('Training GaussianGenerator')
    generator = GaussianGenerator()
    generator = generator.fit(X, Y)

    # Plotting
    plot_sample(generator)
    plot_class_samples(generator)
