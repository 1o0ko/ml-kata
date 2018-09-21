from itertools import chain


class Corpus(object):
    ''' Corpus object '''
    def __init__(self, sentences):
        self._sentences = [
            s.split() if isinstance(s, str) else s for s in sentences]
        self._words = list(chain.from_iterable(self._sentences))

    def words(self):
        return self._words

    def sents(self):
        return self._sentences

    def __add__(self, other):
        return Corpus(chain(self.sents(), other.sents()))


simple = Corpus([
    'this is a sentence',
    'this sentence is long',
    'fishes like to swim',
    'Brown corpus is a resource',
    'I like listening to music',
    'I like reading music books',
    'this is a really long sentence'
])
