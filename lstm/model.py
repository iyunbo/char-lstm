import hashlib
import logging as log
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lstm import preprocessing as prep

on_gpu = torch.cuda.is_available()
version = 'v1'


class CharLSTM(nn.Module):

    def __init__(self, text, n_hidden=512, n_layers=3,
                 drop_prob=0.5, lr=0.001):
        super().__init__()
        self.id = hashlib.sha1(text.encode('utf-8')).hexdigest()
        self.model_file_name = '{}-{}.net'.format(self.id, version)
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars, self.encoded, self.int2char, self.char2int = prep.tokenize(text)

        # creating LSTM layers
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        # Define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # Define the fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        """ Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. """

        # Get the outputs and the new hidden state from the lstm
        x, hidden = self.lstm(x, hidden)

        # Pass through a dropout layer
        x = self.dropout(x)

        # Stack up LSTM outputs using view
        x = x.contiguous().view(-1, self.n_hidden)

        # Pass through the fully-connected layer
        output = self.fc(x)

        # return the final output and the hidden state
        return output, hidden

    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

    def checkpoint(self):

        cp = {'n_hidden': self.n_hidden,
              'n_layers': self.n_layers,
              'chars': self.chars,
              'int2char': self.int2char,
              'char2int': self.char2int,
              'drop_prob': self.drop_prob,
              'lr': self.lr,
              'state_dict': self.state_dict()}

        with open(self.model_file_name, 'wb') as f:
            torch.save(cp, f)

    def load(self, filename=None):
        if not filename:
            filename = self.model_file_name
        log.info("loading previously trained model: {}".format(filename))
        checkpoint = torch.load(filename)
        self.n_hidden = checkpoint['n_hidden']
        self.n_layers = checkpoint['n_layers']
        self.chars = checkpoint['chars']
        self.int2char = checkpoint['int2char']
        self.char2int = checkpoint['char2int']
        self.drop_prob = checkpoint['drop_prob']
        self.lr = checkpoint['lr']
        self.load_state_dict(checkpoint['state_dict'])

    def predict(self, char, hidden=None, top_k=None):
        """ Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        """

        # tensor inputs
        x = np.array([[self.char2int[char]]])
        x = prep.one_hot_encode(x, len(self.chars))
        inputs = torch.from_numpy(x)

        if on_gpu:
            inputs = inputs.cuda()

        # detach hidden state from history
        hidden = tuple([e.data for e in hidden])
        # get the output of the model
        out, hidden = self.forward(inputs, hidden)

        # get the character probabilities
        p = F.softmax(out, dim=1).data
        if on_gpu:
            p = p.cpu()  # move to cpu

        # get top characters
        if top_k is None:
            top_ch = np.arange(len(self.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p / p.sum())

        # return the encoded value of the predicted char and the hidden state
        return self.int2char[char], hidden

    def already_trained(self):
        return os.path.isfile(self.model_file_name)

    def generate(self, size, prime='the', top_k=None):

        if on_gpu:
            self.cuda()
        else:
            self.cpu()

        self.eval()  # eval mode

        # string to next_char array
        chars = [ch for ch in prime]
        hidden = self.init_hidden(1)
        next_char = ''
        for ch in prime:
            next_char, hidden = self.predict(ch, hidden, top_k=top_k)

        chars.append(next_char)

        # Now pass in the previous character and get a new one
        for ii in range(size - len(prime) - 1):
            next_char, hidden = self.predict(chars[-1], hidden, top_k=top_k)
            chars.append(next_char)

        return ''.join(chars)


def train(model, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    """ Training a network

        Arguments
        ---------

        model: CharRNN network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss

    """
    start_time = time.time()
    log.info("training model within {} epochs with batch_size={}, seq_length={}, lr={}, clip={}, val_frac={}, "
             "print_every={}".format(epochs, batch_size, seq_length, lr, clip, val_frac, print_every))
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    data = model.encoded
    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if on_gpu:
        model.cuda()

    counter = 0
    n_chars = len(model.chars)
    val_losses = [-10]
    val_loss_min = 10
    for e in range(epochs):
        # initialize hidden state
        h = model.init_hidden(batch_size)

        for x, y in prep.get_batches(data, batch_size, seq_length):
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = prep.one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            model.zero_grad()

            # get the output from the model
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size * seq_length).long())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for inputs, label in prep.get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = prep.one_hot_encode(inputs, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(label)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if on_gpu:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size * seq_length).long())

                    val_losses.append(val_loss.item())

                model.train()  # reset to train mode after iterating through validation data

                val_loss_avg = np.mean(val_losses)
                if val_loss_avg < val_loss_min:
                    val_loss_min = val_loss_avg
                    model.checkpoint()
                log.info(
                    "Epoch-{:03d}/{:03d}...Step-{:05d} "
                    "==> Loss: {:.4f}...Val Loss: {:.4f} (lowest: {:.4f})".format(e + 1,
                                                                                  epochs,
                                                                                  counter,
                                                                                  loss.item(),
                                                                                  val_loss_avg,
                                                                                  val_loss_min)
                )
    duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    log.info("Training duration: ".format(duration))
    return np.mean(val_losses)
