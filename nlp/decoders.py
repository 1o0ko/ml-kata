import heapq

from data import corpora
from nlp.language_models import TrigramModel

from typing import List, Optional

END = None


def greedy(
        lm,
        seed: List[str],
        end_token: str = END,
        limit: Optional[int] = None):
    '''
    Geedily Finish the sentence given the seed
    '''
    text, prefix_len = seed, len(seed)
    while True:
        probs = lm.conditional_probs(text)
        if not probs:
            break

        next_word, _ = max(probs, key=lambda x: x[1])
        if next_word == END:
            break

        text.append(next_word)
        if limit and len(text) - prefix_len >= limit:
            break

    return text[prefix_len:], lm(text[prefix_len:])


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
        lm,
        seed: List[str],
        end_token: str = END,
        limit: Optional[int] = None,
        beam_width: int = 10):
    '''
    Finish the sentence given the seed
    '''
    # currently we're not using smoothing
    if seed not in lm:
        return (seed, 0.0)

    prefix_len = len(seed)
    prev_beam = Beam(beam_width)
    prev_beam.add(1.0, False, seed)

    while True:
        curr_beam = Beam(beam_width)

        for (sequence_prob, complete, sequence) in prev_beam:
            if complete:
                curr_beam.add(sequence_prob, True, sequence)
            else:
                for next_word, next_prob in lm.conditional_probs(sequence):
                    if next_word == end_token:
                        curr_beam.add(
                            sequence_prob * next_prob, True, sequence)
                    else:
                        curr_beam.add(
                            sequence_prob * next_prob, False, sequence + [next_word])

        (best_prob, best_complete, best_sequence) = max(curr_beam)
        if best_complete or len(best_sequence) - prefix_len == limit:
            return (best_sequence[prefix_len:], best_prob)

        prev_beam = curr_beam


if __name__ == '__main__':

    lm = TrigramModel()
    lm.fit(corpora.simple)

    print(greedy(lm, seed=[None, None], limit=None))
    print(greedy(lm, seed=['I']))
    print(greedy(lm, seed=['I', 'like'], limit=3))
    print(greedy(lm, seed=['I', 'like', 'trains'], limit=3))

    print(beamsearch(lm, seed=[None, None], limit=None))
    print(beamsearch(lm, seed=['I']))
    print(beamsearch(lm, seed=['I', 'like'], limit=3))
    print(beamsearch(lm, seed=['I', 'like', 'trains'], limit=3))
