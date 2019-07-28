'''
The example is kinda stupid, as with a static matrix, we have implicitly
encoded the notion of observation independence.
'''
from math import log
import numpy as np


def greedy_decoder(data):
    # find indices of max value
    indices = np.argmax(data, axis=1)

    # fetch probabilities
    probs = data[(np.arange(data.shape[0]), indices)]

    # calcualte the score
    score = np.prod(-np.log(probs))

    return indices.tolist(), score


def beam_search_decoder(data, k, verbose=False):
    # start with an empty beam
    beam = [(list(), 1.0)]

    # walk over each step in the data
    for word_probs in data:
        all_candidates = list()
        # expand each current candidate
        for (seq, score) in beam:
            for idx, prob in enumerate(word_probs):
                all_candidates.append((seq + [idx], score * -log(prob)))

        # order all candidates by score
        best = sorted(all_candidates, key=lambda tup: tup[1])

        # select k best
        beam = best[:k]
        if verbose:
            print("Best beams:")
            for i, (seq, score) in enumerate(beam):
                print(f"\t{i}) {str(seq):30s}, {score:0.4f}")

    return beam


def main():
    # define a sequence of 10 words over a vocab of 5 words T x |V|.
    data = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1]])

    print("Greedy decoder:")
    print(greedy_decoder(data))

    print("Beam search:")
    print(beam_search_decoder(data, 3, verbose=True))


if __name__ == '__main__':
    import sys
    sys.exit(main())
