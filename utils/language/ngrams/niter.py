import collections

from typing import Iterator, Tuple
from itertools import tee, islice


def consume(iterator, n=None):
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def bigrams(iterable: Iterator[str]) -> Iterator[Tuple[str, str]]:
    ''' Create bi-gram form an iterable
    i -> (i0, i1), (i1, i2), (i2, i3), ...

       > for bigram in bigrams(['all', 'this', 'happened', 'more', 'or', 'less']):
    ...:     print(bigram)
    ...:

    ('all', 'this')
    ('this', 'happened')
    ('happened', 'more')
    ('more', 'or')
    ('or', 'less')
    '''
    a, b = tee(iterable)
    consume(b, 1)
    return zip(a, b)


def ngrams(iterable: Iterator[str], n: int) -> Iterator[Tuple[str, ...]]:
    ''' Create n-grams form an interable
    i -> (i_0, i_1, ..., i_n), (i_1, i_2, ..., i_n+1),  ...

    Example:
       > for ngram in ngrams(['all', 'this', 'happened', 'more', 'or', 'less'], 3):
    ...:     print(ngram)

    ('all', 'this', 'happened')
    ('this', 'happened', 'more')
    ('happened', 'more', 'or')
    ('more', 'or', 'less')
    '''
    iters = tee(iterable, n)
    for skip, i in enumerate(iters):
        consume(i, skip)

    return zip(*iters)
