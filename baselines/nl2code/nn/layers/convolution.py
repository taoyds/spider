# -*- coding: utf-8 -*-

from .core import Layer
from nn.utils.theano_utils import *
import nn.initializations as initializations
import nn.activations as activations
from theano.tensor.nnet import conv
from theano.tensor.signal import pool


class Convolution2d(Layer):
    """a convolutional layer with max pooling"""

    def __init__(self, max_sent_len, word_embed_dim, filter_num, filter_window_size,
                 border_mode='valid', activation='relu', name='Convolution2d'):
        super(Convolution2d, self).__init__()

        self.init = initializations.get('uniform')
        self.activation = activations.get(activation)
        self.border_mode = border_mode

        self.W = self.init((filter_num, 1, filter_window_size, word_embed_dim), scale=0.01, name='W')
        self.b = shared_zeros((filter_num), name='b')

        self.params = [self.W, self.b]

        if self.border_mode == 'valid':
            self.ds = (max_sent_len - filter_window_size + 1, 1)
        elif self.border_mode == 'full':
            self.ds = (max_sent_len + filter_window_size - 1, 1)

        if name is not None:
            self.set_name(name)

    def __call__(self, X):
        # X: (batch_size, max_sent_len, word_embed_dim)

        # valid: (batch_size, nb_filters, max_sent_len - filter_window_size + 1, 1)
        # full: (batch_size, nb_filters, max_sent_len + filter_window_size - 1, 1)
        conv_output = conv.conv2d(X.reshape((X.shape[0], 1, X.shape[1], X.shape[2])),
                                  filters=self.W,
                                  filter_shape=self.W.shape.eval(),
                                  border_mode=self.border_mode)

        output = self.activation(conv_output + self.b.dimshuffle(('x', 0, 'x', 'x')))

        # (batch_size, nb_filters, 1, 1)
        output = pool.pool_2d(output, ds=self.ds, ignore_border=True, mode='max')
        # (batch_size, nb_filters)
        output = output.flatten(2)
        return output
