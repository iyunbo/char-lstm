import logging as log
import os
import unittest

from lstm import model
from lstm.model import CharLSTM, train

sample_text = "hello world! shall we begin? let's go."
trained_model = CharLSTM("hello world! shall we begin? let's go.")
loss = train(trained_model, epochs=2, batch_size=2, seq_length=2, print_every=4)


class PreProcessingTest(unittest.TestCase):

    def test_gpu_should_be_available(self):
        self.assertTrue(model.on_gpu)

    def test_should_create_LSTM_by_default(self):
        net = CharLSTM("hello")
        self.assertEqual(0.001, net.lr)
        self.assertEqual(4, len(net.char2int))
        self.assertEqual(net.n_hidden, net.fc.in_features)
        self.assertEqual(4, net.lstm.input_size)
        self.assertEqual(512, net.lstm.hidden_size)
        self.assertEqual(net.n_layers, net.lstm.num_layers)
        self.assertEqual(0.5, net.lstm.dropout)
        log.info("LSTM layers: {}".format(net.lstm))

    def test_should_train_network(self):
        self.assertTrue(loss > 0)

    def test_should_train_network_on_cpu(self):
        model.on_gpu = False
        net = CharLSTM(sample_text)
        val_loss = train(net, epochs=2, batch_size=2, seq_length=2, print_every=4)
        self.assertTrue(val_loss > 0)

    def test_should_save_model_on_checkpoint(self):
        trained_model.checkpoint()
        self.assertTrue(os.path.isfile(trained_model.model_file_name))

    def test_should_predict_next_char(self):
        h = trained_model.init_hidden(1)
        char, _ = trained_model.predict('e', h)
        log.info("next char: {}".format(char))
        self.assertTrue(char)

    def test_should_predict_next_char_with_topk(self):
        h = trained_model.init_hidden(1)
        char, _ = trained_model.predict('e', h, top_k=2)
        log.info("next char: {}".format(char))
        self.assertTrue(char)

    def test_should_generate_text_for_given_size(self):
        text = trained_model.generate(50, "all")
        log.info(text)
        self.assertEqual(50, len(text))

    def test_should_load_from_file(self):
        n_hidden = trained_model.n_hidden
        trained_model.load()
        self.assertEqual(n_hidden, trained_model.n_hidden)
