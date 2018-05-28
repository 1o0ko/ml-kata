import abc
import operator
import random

from collections import Counter, defaultdict
from functools import reduce

from nltk.corpus import reuters
from utils.language import trigrams


class LanguageModel(abc.ABC):
    def __call__(self, sentence):
        return self.prob(sentence)

    @abc.abstractmethod
    def fit(self, corpus):
        pass

    @abc.abstractmethod
    def sample(self, n=10):
        pass

    @abc.abstractmethod
    def _prob(self, sentence):
        pass

    def prob(self, sentence):
        if not isinstance(sentence, list) and isinstance(sentence, str):
            sentence = sentence.split()
        else:
            raise ValueError("Not a sting or list!")

        return self._prob(sentence)


class UnigramModel(LanguageModel):

    def fit(self, corupus):
        self.model = Counter(corupus.words())
        self.total_count = len(corupus.words())

        for word in self.model:
            self.model[word] /= float(self.total_count)

    def sample(self, n=10):
        return ' '.join([self.sample_one_word() for _ in range(n)])

    def _prob(self, sentence):
        return reduce(operator.mul, [self.prob_word(word)
                                     for word in sentence])

    def prob_word(self, word):
        return self.model.get(word, min(self.model.values()))

    def sample_one_word(self):
        r = random.random()
        accumulator = .0
        for word, freq in self.model.items():
            accumulator += freq
            if accumulator >= r:
                return word


class TriGramModel(LanguageModel):

    def __init__(self):
        self.unigram_lm = UnigramModel()

    def fit(self, corpus):
        # for unigram backoff
        self.unigram_lm.fit(corpus)

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

    def _prob(self, sentence):
        ''' P([w1, w2, ..., wN])

        Naive probability of a sentence with an unigram back-off
        '''
        probs = [
            self.prob_trigram(t) for t in trigrams(sentence, pad_left=True)]

        return reduce(operator.mul, probs)

    def prob_trigram(self, trigram):
        ''' P(t_3 | t_2, t1) '''
        (t1, t2, t3) = trigram
        items = self.model.get((t1, t2), None)
        if items and t3 in items:
            return items[t3]

        # naive backoff maybe char level LM?
        return self.unigram_lm(t3)


if __name__ == '__main__':
    print('Training a LM')
    trigram_lm = TriGramModel()
    trigram_lm.fit(reuters)

    print('A sample from the model')
    print(trigram_lm.sample())

    p_1 = trigram_lm('This is a sentence')
    p_2 = trigram_lm('Fishes like brown music')

    assert p_1 > p_2
