import abc
import random
import numpy as np

from collections import Counter, defaultdict
from itertools import chain

from nlp.ngrams import trigrams


class Corpus(object):
    ''' Corpus object '''
    def __init__(self, sentences):
        self._sentences = [
            s.split() if isinstance(s, str) else s for s in sentences]
        self._words = list(chain.from_iterable(self._sentences))

    @property
    def words(self):
        return self._words

    @property
    def sents(self):
        return self._sentences

    def __add__(self, other):
        return Corpus(chain(self.sents, other.sents))


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
    def probs(self, sentence):
        ''' Returns the list of tokens and their probabilities

        [(t_1, p(t_1)), ... , (t_N, p(t_N))]
        '''

        pass

    def __check(self, sentence):
        if isinstance(sentence, str):
            sentence = sentence.split()
        elif not isinstance(sentence, list):
            raise ValueError("Not a sting or list!")
        return sentence

    def prob(self, sentence):
        ''' Calculates probability of the sentence W = (w_1, ..., w_N)

        P(W) = P((w_1, w_2, ..., w_N))

        Uses log-probabilities
        '''
        return np.exp(np.sum(np.log(
            [p for (t, p) in self.probs(self.__check(sentence))]
        )))

    def perplexity(self, sentence):
        ''' Calculates perplexity of the sentence W = (w_1, ..., w_N)

        PP(W) = P((w_1, w_2, ..., w_N)) ^ (- 1 / N)
        '''
        sentence = self.__check(sentence)
        return np.power(1 / self.prob(sentence), 1 / len(sentence))


class UnigramModel(LanguageModel):

    def fit(self, corpus):
        self.model = Counter(corpus.words)
        self.total_count = len(corpus.words)

        for word in self.model:
            self.model[word] /= float(self.total_count)

        self.min_prob = min(self.model.values())

    def sample(self, n=10):
        return ' '.join([self.sample_one_word() for _ in range(n)])

    def probs(self, sentence):
        '''Returns list of tuples of (word, it's probability)'''
        return [(word, self.prob_word(word)) for word in sentence]

    def prob_word(self, word):
        ''' calculate the probability of a word '''
        return self.model.get(word, self.min_prob)

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
        # for unigram back-off
        self.unigram_lm.fit(corpus)

        # dict of dicts
        self.model = defaultdict(lambda: defaultdict(float))

        for sentence in corpus.sents:
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
        Greedy sampling from a trigram model
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

    def probs(self, sentence):
        ''' List of tuples: trigram, it's probability with unigram back-off

        [
         ((w_1,     w_2,     w_3), P(w_3 | w_2, w_1)), ... ,
         ((w_(N-2), w_(N-1), w_N), P(w_3 | w_2, w_1))
        ]
        '''
        return [(t, self.prob_trigram(t))
                for t in trigrams(sentence, pad_left=True)]

    def prob_trigram(self, trigram):
        ''' P(t_3 | t_2, t_1) with unigram back-off
        '''
        (t1, t2, t3) = trigram
        items = self.model.get((t1, t2), None)
        if items and t3 in items:
            return items[t3]

        return self.unigram_lm(t3)


if __name__ == '__main__':
    corpus = Corpus([
        'this is a sentence',
        'this sentence is long',
        'fishes like to swim',
        'Brown corpus is a resource',
        'I like listening to music',
        'this is a really long sentence'
    ])

    print('Training a LM')
    trigram_lm = TriGramModel()
    trigram_lm.fit(corpus)

    print('A sample from the model')
    print(trigram_lm.sample())

    hi_p_sent = 'This is a sentence'
    lo_p_sent = 'Fishes like brown music'
    p_1 = trigram_lm(hi_p_sent)
    p_2 = trigram_lm(lo_p_sent)

    print(f"{hi_p_sent}: {p_1}, {lo_p_sent}: {p_2}")
    assert p_1 > p_2
