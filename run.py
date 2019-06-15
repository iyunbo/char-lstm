import logging as log

from lstm import preprocessing as prep
from lstm.model import CharLSTM, train


def main():
    text = prep.load_data()
    network = CharLSTM(text, n_hidden=512, n_layers=3)
    train(network, epochs=40, batch_size=128, seq_length=120, print_every=50)
    generated_text = network.generate(1500, prime='And Sally said')
    log.info("generated:")
    log.info(generated_text)


if __name__ == "__main__":
    main()
