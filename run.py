import argparse
import logging as log
import os.path as path

from lstm import preprocessing as prep
from lstm.model import CharLSTM, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="the text file to train on", default='anna.txt')
    parser.add_argument("-p", "--prime", help="the prime message to begin with for generation", required=True)
    parser.add_argument("-n", "--count", help="the number of characters of generated text", type=int, required=True)
    parser.add_argument("-s", "--seq_length", help="the text file to train on", type=int, default=120)
    args = parser.parse_args()
    file = args.file
    prime = args.prime
    count = args.count
    seq_length = args.seq_length

    log.info("generating text of size {} from file {} with prime '{}' ...".format(count, file, prime))

    text = prep.load_data(path=path.join('data', file))

    network = CharLSTM(text, n_hidden=512, n_layers=3)
    if network.already_trained():
        network.load()
    else:
        train(network, epochs=40, batch_size=128, seq_length=seq_length, print_every=50)

    log.info("generating text now:")
    generated_text = network.generate(count, prime=prime)
    print(generated_text)


if __name__ == "__main__":
    main()
