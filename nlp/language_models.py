import abc
import random
import numpy as np

from collections import Counter, defaultdict
from itertools import chain
from typing import List, Optional

from nlp.ngrams import ngrams

Token = Optional[str]
Seed = Optional[List[Token]]


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
    def sample(self, k=10):
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

    def sample(self, k=10):
        return ' '.join([self.sample_one_word() for _ in range(k)])

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


class NgramModel(LanguageModel):
    '''
    Builds a n-gram language model

    Args:
        n - order of the n-gram
    '''
    def __init__(self, n: int):
        if n <= 1:
            raise ValueError("n should be greater than 1")

        self.unigram_lm = UnigramModel()
        self.model = defaultdict(lambda: defaultdict(float))
        self.n = n

    def fit(self, corpus):
        # for unigram back-off
        self.unigram_lm.fit(corpus)

        for sentence in corpus.sents:
            for *hist, word in ngrams(sentence, self.n, pad_right=True, pad_left=True):
                self.model[tuple(hist)][word] += 1

        # Let's transform the counts to probabilities
        # TODO: add smoothing
        for hist in self.model:
            total_count = float(sum(self.model[hist].values()))
            for word in self.model[hist]:
                self.model[hist][word] /= total_count

    def sample(self,
               text: Optional[List[Token]] = None,
               k: Optional[int] = None):
        ''' Greedy sampling from a ngram model '''
        # Initialize context
        text = text if text else [None] * (self.n - 1)
        sentence_finished = False

        while not sentence_finished:
            r = random.random()
            accumulator = .0

            history = tuple(text[-(self.n - 1):])
            for word, freq in self.model[history].items():
                accumulator += freq

                if accumulator >= r:
                    text.append(word)
                    break

            if k and len(text[self.n - 1:]) == k:
                text.append(".")
                sentence_finished = True

            if text[-(self.n - 1):] == [None] * (self.n - 1):
                sentence_finished = True

        return ' '.join([t for t in text if t])

    def prob_ngram(self, ngram):
        ''' P(t_n | t_(n-1), ..., t_1) with unigram back-off
        '''
        *hist, word = ngram
        probs = self.model.get(tuple(hist), None)
        if probs and word in probs:
            return probs[word]

        return self.unigram_lm(word)

    def probs(self, sentence):
        ''' List of tuples: trigram, it's probability with unigram back-off

        [
         ((w_1,        w_2,        ..., w_n), P(w_n | w_(n-1), ..., w_1)), ... ,
         ...
         ((w_(N-(n-1), w_(N-(n-2), ..., w_N), P(w_n | w_(n-1), ..., w_1)), ... ,
        ]
        '''
        return [(n, self.prob_ngram(n))
                for n in ngrams(sentence, self.n, pad_left=True)]


class BigramModel(NgramModel):
    ''' Convenience wrapper '''
    def __init__(self):
        super().__init__(2)


class TrigramModel(NgramModel):
    ''' Convenience wrapper '''
    def __init__(self):
        super().__init__(3)


if __name__ == '__main__':
    corpus = Corpus([
        'this is a sentence',
        'this sentence is long',
        'fishes like to swim',
        'Brown corpus is a resource',
        'I like listening to music',
        'I like reading music books',
        'this is a really long sentence'
    ])

    print('Training unigram  LM')
    unigram_lm = UnigramModel()
    unigram_lm.fit(corpus)
    print('\tA sample from the model:', unigram_lm.sample())

    print('Training a bigram LM')
    bigram_lm = BigramModel()
    bigram_lm.fit(corpus)

    print('\tA sample from the model:', bigram_lm.sample(k=3))
    print('\tA sample from the model:', bigram_lm.sample(text=['I']))

    print('Training a 3-gram LM')
    trigram_lm = TrigramModel()
    trigram_lm.fit(corpus)

    print('\tA sample from the model:', trigram_lm.sample())
    print('\tA sample from the model:', trigram_lm.sample(text=[None, 'I']))

    hi_p_sent = 'This is a sentence'
    lo_p_sent = 'Fishes like brown music'
    p_1 = trigram_lm(hi_p_sent)
    p_2 = trigram_lm(lo_p_sent)

    print(f"{hi_p_sent}: {p_1}, {lo_p_sent}: {p_2}")
    assert p_1 > p_2
