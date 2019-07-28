import heapq
import numpy as np

from data import corpora
from nlp.language_models import TrigramModel, Token, Seed, START, END

from typing import Callable, List, Optional, Tuple, Union

TokenScore = Tuple[Token, float]
SequenceScore = Tuple[List[Token], float]


def greedy(
    prob_fn: Callable[[List[Token]], List[TokenScore]],
    seed: List[Token],
    end_token: Token = END,
    limit: Optional[int] = None) -> SequenceScore:
    '''
    Geedily finish the sentence given the seed and probability function.
    '''
    text, prefix_len = seed, len(seed)
    text_probs = []
    while True:
        probs = prob_fn(text)
        next_word, prob = max(prob_fn(text), key=lambda x: x[1])
        if next_word == END:
            text_probs.append(1e-20); break

        text.append(next_word)
        text_probs.append(prob)
        if limit and len(text) - prefix_len >= limit:
            break

    # Use sum-log trick for numerical stability.
    text_score = np.exp(np.sum(np.log(text_probs)))

    return text[prefix_len:], text_score


class Beam(object):
    """
    For comparison of prefixes, the tuple (prefix_prob, complete_sentence) is uesd.
    This is so that if two prefixes have eequal probabilities then a complete sentence is preffered
    over and incomplete one since (0.5, False) < (0.5, True)
    """

    def __init__(self, beam_width):
        self.heap = []
        self.beam_width = beam_width

    def add(self, prob, complete, prefix):
        heapq.heappush(self.heap, (prob, complete, prefix))

        # if we exeed the beam size, remove the "worst" item
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)

    def __repr__(self):
        return str(self.heap)


def beamsearch(
    prob_fn: Callable[[List[Token]], List[TokenScore]],
    seed: List[Token],
    end_token: Token = END,
    limit: Optional[int] = None,
    beam_width: int = 10) -> SequenceScore:
    '''
    Finish the sentence given the seed
    '''
    prefix_len = len(seed)
    prev_beam = Beam(beam_width)
    prev_beam.add(1.0, False, seed)

    while True:
        curr_beam = Beam(beam_width)

        for (sequence_prob, complete, sequence) in prev_beam:
            if complete:
                curr_beam.add(sequence_prob, True, sequence)
            else:
                for next_word, next_prob in prob_fn(sequence):
                    if next_word == end_token:
                        curr_beam.add(sequence_prob * next_prob, True, sequence)
                    else:
                        curr_beam.add(sequence_prob * next_prob, False, sequence + [next_word])

            best_prob, best_complete, best_sequence = max(curr_beam)

        if best_complete or len(best_sequence) - prefix_len == limit:
            return (best_sequence[prefix_len:], best_prob)

        prev_beam = curr_beam


if __name__ == '__main__':

    lm = TrigramModel()
    lm.fit(corpora.simple)

    prob_fn = lambda text: lm.conditional_probs(text)
    print("Greedy.")
    print(greedy(prob_fn, seed=[START, START], limit=None))
    print(greedy(prob_fn, seed=['I']))
    print(greedy(prob_fn, seed=['I', 'like'], limit=3))
    print(greedy(prob_fn, seed=['I', 'like', 'trains'], limit=3))

    print("Beam search.")
    print(beamsearch(prob_fn, seed=[START, START], limit=None))
    print(beamsearch(prob_fn, seed=['I']))
    print(beamsearch(prob_fn, seed=['I', 'like'], limit=3))
    print(beamsearch(prob_fn, seed=['I', 'like', 'trains'], limit=3))
