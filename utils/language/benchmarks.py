import random
import time
import sys

from utils.language.ngrams import niter, nnltk

SEQUENCE = 'this is the best day of my life so far and I love it'.split()


class MyTimer():

    def __init__(self):
        self.start = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.start
        msg = 'The function took {time} seconds to complete'
        print(msg.format(time=runtime))


def make_ngrams(fn, n, n_times):
    for i in range(1, n + 1):
        for j in range(n_times):
            _ = [ngram for ngram in fn(SEQUENCE, i)]


def main():
    with MyTimer():
        make_ngrams(niter.ngrams, 5, 10000)

    with MyTimer():
        make_ngrams(nnltk.ngrams, 5, 10000)


if __name__ == '__main__':
    sys.exit(main())
