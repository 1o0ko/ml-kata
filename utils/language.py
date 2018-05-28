from itertools import chain


def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    ''' padding (from NLTK)'''
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n-1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n-1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    ''' Create ngrams (from NLTK)'''
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                        left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def trigrams(sequence, **kwargs):
    ''' Return trigrams generated from a sequence '''
    for item in ngrams(sequence, 3, **kwargs):
        yield item
