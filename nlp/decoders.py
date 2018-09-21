from data import corpora
from nlp.language_models import TrigramModel
from nltk.corpus import brown

from typing import List, Optional

START = '<START>'
END = '<END>'


def greedy(
        probabilities_function,
        seed: List[str] =[START, START],
        end_token: str = END,
        limit: Optional[int] = None):
    '''
    Finish the sentence given the seed
    '''
    return seed


def beamsearch(
        probabilities_function,
        seed: List[str] =[START, START],
        end_token: str = END,
        limit: Optional[int] = None,
        beam_width: int = 10):
    '''
    Finish the sentence given the seed
    '''
    return seed


if __name__ == '__main__':

    lm = TrigramModel()
    lm.fit(corpora.simple)

    print(beamsearch(lm.probs, limit=None))
    print(beamsearch(lm.probs, seed=['the', 'man'], limit=3))
