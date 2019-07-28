import unittest
import numpy as np

from nlp import beam_search_numpy

DATA = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                 [0.5, 0.4, 0.3, 0.2, 0.1],
                 [0.1, 0.2, 0.3, 0.4, 0.5],
                 [0.5, 0.4, 0.3, 0.2, 0.1],
                 [0.1, 0.2, 0.3, 0.4, 0.5],
                 [0.5, 0.4, 0.3, 0.2, 0.1],
                 [0.1, 0.2, 0.3, 0.4, 0.5],
                 [0.5, 0.4, 0.3, 0.2, 0.1],
                 [0.1, 0.2, 0.3, 0.4, 0.5],
                 [0.5, 0.4, 0.3, 0.2, 0.1]])


class TestGreedy(unittest.TestCase):
    def setUp(self):
        self.data = DATA

    def test_greedy_decoder(self):
        seq, p = [4, 0, 4, 0, 4, 0, 4, 0, 4, 0], 0.025600863289563108
        seq_beam, p_beam= beam_search_numpy.greedy_decoder(self.data)
        self.assertEqual(p, p_beam)
        self.assertSequenceEqual(seq, seq_beam)


if __name__ == '__main__':
    unittest.main()
