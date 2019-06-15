import logging as log
import unittest

import numpy as np
from mock import patch, mock_open

from lstm import preprocessing as prep


class PreProcessingTest(unittest.TestCase):

    @patch("builtins.open", mock_open(read_data="hello foo bar!"))
    def test_load_data(self):
        path = "data path"
        text = prep.load_data(path)
        log.info("text: {}".format(text))
        self.assertEqual(text, "hello foo bar!")

    def test_tokenize(self):
        text = "hello Bob!"
        _, tokens, _, _ = prep.tokenize(text)
        self.assertEqual(len(tokens), len(text))
        self.assertEqual(tokens[2], tokens[3])
        self.assertEqual(tokens[4], tokens[7])

    def test_tokenized_mapping(self):
        text = "hello Bob!"
        _, _, int2char, char2int = prep.tokenize(text)
        h_int = char2int['h']
        one_char = int2char[1]
        self.assertEqual(int2char[h_int], 'h')
        self.assertEqual(char2int[one_char], 1)

    def test_one_hot(self):
        test_seq = np.array([[3, 5, 1]])
        one_hot = prep.one_hot_encode(test_seq, 8)
        self.assertTrue(np.array_equal(one_hot, np.array(
            [[[0., 0., 0., 1., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 1., 0., 0.],
              [0., 1., 0., 0., 0., 0., 0., 0.]]]
        )))

    def test_get_batches(self):
        encoded = np.array([1., 2., 3., 1., 2., 3., 1., 2., 2., 3., 4., 5., 2., 3., 4., 5.,
                            3., 4., 5., 6., 7., 3., 4., 5., 4., 5., 6., 7., 8., 9., 0., 1.])
        batches = prep.get_batches(encoded, 2, 4)
        x, y = next(batches)
        self.assertTrue(np.array_equal(
            np.array([[1., 2., 3., 1.],
                      [3., 4., 5., 6.]]),
            x
        ))
        self.assertTrue(np.array_equal(
            np.array([[2., 3., 1., 2.],
                      [4., 5., 6., 7.]]),
            y
        ))

        x, y = next(batches)
        self.assertTrue(np.array_equal(
            np.array([[2., 3., 1., 2.],
                      [7., 3., 4., 5.]]),
            x
        ))
        self.assertTrue(np.array_equal(
            np.array([[3., 1., 2., 2.],
                      [3., 4., 5., 4.]]),
            y
        ))
