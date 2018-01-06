# Beam Search

from nltk.corpus import reuters
from nltk import trigrams
from collections import Counter, defaultdict

import random


class LanguageModel(object):

    def fit(self, corpus):
        pass

    def sample(self, n=10):
        pass


class UnigramModel(LanguageModel):

    def fit(self, corupus):
        self.model = Counter(corupus.words())
        self.total_count = len(corupus.words())

        for word in self.model:
            self.model[word] /= float(self.total_count)

    def sample(self, n=10):
        return ' '.join([self.sample_one_word() for _ in range(n)])

    def sample_one_word(self):
        r = random.random()
        accumulator = .0
        for word, freq in self.model.items():
            accumulator += freq
            if accumulator >= r:
                return word


class TriGramModel(LanguageModel):
    def fit(self, corpus):
        # dict of dicts
        self.model = defaultdict(lambda: defaultdict(lambda: 0))

        for sentence in corpus.sents():
            for w1, w2, w3 in trigrams(
                    sentence, pad_right=True, pad_left=True):
                self.model[(w1, w2)][w3] += 1

        # Let's transform the counts to probabilities
        for w1_w2 in self.model:
            total_count = float(sum(self.model[w1_w2].values()))
            for w3 in self.model[w1_w2]:
                self.model[w1_w2][w3] /= total_count

    def sample(self, n=None):
        text = [None, None]
        sentence_finished = False

        while not sentence_finished:
            r = random.random()
            accumulator = .0

            for word, freq in self.model[tuple(text[-2:])].items():
                accumulator += freq

                if accumulator >= r:
                    text.append(word)
                    break

            if n and len(text[2:]) == n:
                text.append(".")
                sentence_finished = True

            if text[-2:] == [None, None]:
                sentence_finished = True

        return ' '.join([t for t in text if t])


import heapq


class Beam(object):

    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width

    def add(self, prob, complete, prefix):
        heapq.heappush(self.heap, (prob, complete, prefix))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)


def beamsearch(
        probabilities_function,
        beam_width=10,
        clip_len=-1):
    '''
    Performs beam search.
    '''
    prev_beam = Beam(beam_width)
    prev_beam.add(1.0, False, ['<start>'])
    while True:
        curr_beam = Beam(beam_width)

        # Add complete sentences that do not yet have the best probability to the current beam,
        # the rest prepare to add more words to them.
        for (prefix_prob, complete, prefix) in prev_beam:
            if complete:
                curr_beam.add(prefix_prob, True, prefix)
            else:
                # Get probability of each possible next word for the incomplete
                # prefix.
                for (next_prob, next_word) in probabilities_function(prefix):
                    if next_word == '<end>':  # if next word is the end token then mark prefix as complete and leave out the end token
                        curr_beam.add(prefix_prob * next_prob, True, prefix)
                    else:  # if next word is a non-end token then mark prefix as incomplete
                        curr_beam.add(
                            prefix_prob * next_prob, False, prefix + [next_word])

        (best_prob, best_complete, best_prefix) = max(curr_beam)
        # if most probable prefix is a complete sentence or has a length that
        # exceeds the clip length (ignoring the start token) then return it
        if best_complete or len(best_prefix) - 1 == clip_len:
            # return best sentence without the start token and together with
            # its probability
            return (best_prefix[1:], best_prob)

        prev_beam = curr_beam


if __name__ == '__main__':
    unigram_lm = UnigramModel()
    unigram_lm.fit(reuters)
    unigram_lm.sample(5)

    trigram_lm = TriGramModel()
    trigram_lm.fit(reuters)
    trigram_lm.sample()
