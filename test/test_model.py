import logging as log
import os
import unittest

from lstm import model
from lstm.model import CharLSTM, train


class PreProcessingTest(unittest.TestCase):

    def test_gpu_should_be_available(self):
        self.assertTrue(model.on_gpu)

    def test_should_create_LSTM_by_default(self):
        net = CharLSTM("hello")
        self.assertEqual(0.001, net.lr)
        self.assertEqual(4, len(net.char2int))
        self.assertEqual(net.n_hidden, net.fc.in_features)
        self.assertEqual(5, net.lstm.input_size)
        self.assertEqual(512, net.lstm.hidden_size)
        self.assertEqual(net.n_layers, net.lstm.num_layers)
        self.assertEqual(0.5, net.lstm.dropout)
        log.info("LSTM layers: {}".format(net.lstm))

    def test_should_train_network(self):
        net = CharLSTM("hello world! shall we begin? let's go.")
        val_loss = train(net, epochs=2, batch_size=2, seq_length=2, print_every=4)
        self.assertTrue(val_loss > 0)

    def test_should_train_network_on_cpu(self):
        net = CharLSTM("hello world! shall we begin? let's go.")
        model.on_gpu = False
        val_loss = train(net, epochs=2, batch_size=2, seq_length=2, print_every=4)
        self.assertTrue(val_loss > 0)

    def test_should_save_model_on_checkpoint(self):
        net = CharLSTM("test")
        model.checkpoint(net)
        self.assertTrue(os.path.isfile('char-lstm-{}.net'.format(model.version)))