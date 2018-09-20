import collections

from typing import Iterator, Tuple, Optional
from itertools import tee, islice, chain


def consume(iterator, n=None):
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    ''' padding (from NLTK)'''
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(iterable: Iterator[str], n: int,
           pad_left: bool = False,
           pad_right: bool = False,
           left_pad_symbol: Optional[str] = None,
           right_pad_symbol: Optional[str] = None) -> Iterator[Tuple[str, ...]]:
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
    # pad the sequence
    iterable = pad_sequence(iterable, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)
    # split iterators
    iters = tee(iterable, n)
    for skip, i in enumerate(iters):
        consume(i, skip)

    return zip(*iters)


def bigrams(iterable: Iterator[str], **kwargs) -> Iterator[Tuple[str, str]]:
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
    for item in ngrams(iterable, 2, **kwargs):
        yield item


def trigrams(iterable: Iterator[str], **kwargs) -> Iterator[Tuple[str, str, str]]:
    ''' Return trigrams generated from a sequence '''
    for item in ngrams(iterable, 3, **kwargs):
        yield item
