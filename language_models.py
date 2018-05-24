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
        '''
        Gready sampling from a trigram model
        '''
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



if __name__ == '__main__':
    unigram_lm = UnigramModel()
    unigram_lm.fit(reuters)
    unigram_lm.sample(5)

    trigram_lm = TriGramModel()
    trigram_lm.fit(reuters)
    trigram_lm.sample()
