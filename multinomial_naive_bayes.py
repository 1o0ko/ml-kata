''' Assignment 1.3 MultinomialNaiveBayes '''
import numpy as np


class MultinomialNaiveBayes(object):

    def __init__(self):
        self.trained = False
        self.likelihood = 0
        self.prior = 0

    def train(self, x, y, smooth=False):
        # n_docs = no. of documents
        # n_words = no. of unique words
        n_docs, n_words = x.shape

        # classes = a list of possible classes
        classes = np.unique(y)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words, n_classes))

        prior = np.mean(y == classes, axis=0)
        for cls in classes:
            docs_in_class, _ = np.nonzero(y == cls)
            prior[cls] = 1.0 * len(docs_in_class) / n_docs

            if not smooth:
                likelihood[:, cls] = np.sum(
                    x[docs_in_class], axis=0) / np.sum(x[docs_in_class])
            else:
                likelihood[:, cls] = (
                    np.sum(x[docs_in_class], axis=0) + 1) / (np.sum(x[docs_in_class]) + n_words)

        params = np.zeros((n_words + 1, n_classes))
        for i in range(n_classes):
            params[0, i] = np.log(prior[i])
            params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
