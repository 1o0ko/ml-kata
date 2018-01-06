'''
Minimal Example of  Byte Pair Encoding
Based on Algorithm 1 from 'Neural Machine Translation of Rare Words with Subword Units'
by Rico Sennrich and Barry Haddow and Alexandra Birch (https://arxiv.org/pdf/1508.07909.pdf)
'''
import re
import collections


def get_pairs_counts(vocab):
    '''
    Dictionary of words with their count
    {
        'b i g </w>' : 5,
        'b i g g e r </w>' : 2,
        'b i g g e s t </w': 1,
    }
    '''
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for pair in zip(symbols, symbols[1:]):
            pairs[pair] += freq

    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    '''
     we have to escape special characters like "*" or "?" and
     find 'bigram' that is not preceded by a non-whitespacece (?<!\S)
                        is not followed by a non-whitespace (?!\S
    '''
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        '''
        replace the leftmost non-overlapping occurrences of 'p',
        don't change if 'p' isn't found.
        '''
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out, (' '.join(pair), ''.join(pair))


def compress(vocab, num_merges=10):
    ops = []
    for i in range(num_merges):
        pairs = get_pairs_counts(vocab)
        most_common = max(pairs, key=pairs.get)
        vocab, (old, new) = merge_vocab(most_common, vocab)
        ops.append((old, new))

    return vocab, ops


def encode(word, ops):
    new_word = word[:]
    for (old, new) in ops:
        p = re.compile(r"(?<!\S)%s(?!\S)" % re.escape(old))
        new_word = p.sub(new, new_word)
    return new_word


def decode(word, ops):
    '''
    decoding is applying the transformations in reverse
    '''
    return encode(word, ((b, a) for a, b in reversed(ops)))


if __name__ == '__main__':
    # Toy vocab encoding <-> decoding
    vocab = {
        'l o w </w>': 5,
        'l o w e r </w>': 2,
        'l o w e s t </w>': 1,
        'n e w e s t </w>': 6,
        'w i d e s t </w>': 3
    }

    encoded_voc, ops = compress(vocab)

    # Visualy inspect outcome
    print(encode('l o w e s t </w>', ops))
    print(decode('low est</w>', ops))

    # Check the whole vocabulary
    for word in vocab:
        assert decode(encode(word, ops), ops)
