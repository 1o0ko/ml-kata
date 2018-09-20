'''
Heap-based beam-search decoding for a 3-gram language model
'''

import heapq

from nltk.corpus import reuters
from nltk import trigrams
from collections import Counter, defaultdict

START = '<START>'
END = '<END>'


class TriGramModel(object):
    def fit(self, corpus):
        # dict of dicts
        self.model = defaultdict(lambda: defaultdict(lambda: 0))

        for sentence in corpus.sents():
            for w1, w2, w3 in trigrams(
                sentence + [END], pad_left=True, left_pad_symbol=START):
                self.model[(w1, w2)][w3] += 1

        # Let's transform the counts to probabilities
        for w1_w2 in self.model:
            total_count = float(sum(self.model[w1_w2].values()))
            for w3 in self.model[w1_w2]:
                self.model[w1_w2][w3] /= total_count

    def probs(self, sequence):
        while len(sequence) < 2:
            sequence.insert(0, START)

        w1_w2 = tuple(sequence[-2:])
        if w1_w2 in self.model:
            res = self.model[w1_w2].items()
        else:
            res = [(END, 1.0)]

        return res


class Beam(object):
    """
    For comparison of prefixes, the tuple (prefix_prob, complete_sentence) is uesd.
    This is so that if two prefixes have eequal probabilities then a complete sentence is preffered
    over and incomplete one since (0.5, False) < (0.5, True)
    """

    def __init__(self, beam_width):
        self.heap = []
        self.beam_width = beam_width

    def add(self, prob, complete, prefix):
        heapq.heappush(self.heap, (prob, complete, prefix))

        # if we exeed the beam size, remove the "worst" item
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)

    def __repr__(self):
        return str(self.heap)


def beamsearch(
        probabilities_function,
        seed=[START, START],
        beam_width=10, clip_len=-1,
        end_token=END):
    '''
    Finish the sentence given the seed
    '''
    prev_beam = Beam(beam_width)
    prev_beam.add(1.0, False, seed)
    while True:
        curr_beam = Beam(beam_width)

        for (sequence_prob, complete, sequence) in prev_beam:
            if complete:
                curr_beam.add(sequence_prob, True, sequence)
            else:
                for i, (next_word, next_prob) in enumerate(
                        probabilities_function(sequence)):
                    if next_word == end_token:
                        curr_beam.add(
                            sequence_prob * next_prob, True, sequence)
                    else:
                        curr_beam.add(
                            sequence_prob *
                            next_prob,
                            False,
                            sequence +
                            [next_word])

        (best_prob, best_complete, best_sequence) = max(curr_beam)
        if best_complete or len(best_sequence) - 2 == clip_len:
            return (best_sequence[2:], best_prob)

        prev_beam = curr_beam


if __name__ == '__main__':

    lm = TriGramModel()
    lm.fit(reuters)

    seq, prob = beamsearch(lm.probs, clip_len=-1)
    print(seq, prob, lm.probs(seq))
    print(beamsearch(lm.probs, seed=['the', 'man'], clip_len=-1))
