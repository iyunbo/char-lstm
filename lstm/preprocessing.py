import logging
import sys

import numpy as np

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


def load_data(path):
    # open text file and read in data as `text`
    with open(path, 'r') as f:
        text = f.read()
    logging.debug("data loading done : {} {}".format(text[:100], "..."))
    return text


def tokenize(text):
    # we create two dictionaries:
    # 1. int2char, which maps integers to characters
    # 2. char2int, which maps characters to unique integers
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # encode the text
    tokenized = np.array([char2int[ch] for ch in text])
    return chars, tokenized, int2char, char2int


def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


def get_batches(arr, batch_size: int, seq_length: int):
    """Create a generator that returns batches of size
       batch_size x seq_length from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    """

    batch_length = batch_size * seq_length
    # total number of batches we can make
    batch_count = len(arr) // batch_length

    # Keep only enough characters to make full batches
    arr = arr[:batch_count * batch_length]
    # Reshape into batch_size rows
    arr = arr.reshape(batch_size, -1)

    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n + seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        # assign the first element for the last target
        if n + seq_length < arr.shape[1]:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
        else:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
