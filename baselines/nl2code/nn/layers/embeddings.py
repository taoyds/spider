# -*- coding: utf-8 -*-

from .core import Layer
from nn.utils.theano_utils import *
import nn.initializations as initializations

import nn.activations as activations
from theano.ifelse import ifelse


def get_embed_iter(file_path):
    for line in open(file_path):
        line = line.strip()
        data = line.split(' ')

        word = data[0]
        embed = np.asarray([float(e) for e in data[1:]], dtype='float32')

        yield word, embed


class Embedding(Layer):
    '''
        Turn positive integers (indexes) into denses vectors of fixed size.
        eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    '''
    def __init__(self, input_dim, output_dim, init='uniform', name=None):

        super(Embedding, self).__init__()
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = self.init((self.input_dim, self.output_dim), scale=0.1)
        self.params = [self.W]

        if name is not None:
            self.set_name(name)

    def get_output_mask(self, X):
        return (T.ones_like(X) * (1 - T.eq(X, 0))).astype('int8')

    def init_pretrained(self, file_path, vocab):
        W = self.W.get_value(borrow=True)
        inited_words = set()

        for word, embed in get_embed_iter(file_path):
            if word in vocab:
                idx = vocab[word]
                W[idx] = embed

                inited_words.add(word)

        return inited_words

    def __call__(self, X, mask_zero=False):
        out = self.W[X]
        if mask_zero:
            return out, self.get_output_mask(X)
        else:
            return out


class HybridEmbedding(Layer):
    '''
        Turn positive integers (indexes) into denses vectors of fixed size.
        eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    '''
    def __init__(self, embed_size, unfixed_embed_size, embed_dim, init='uniform', name='HybridEmbedding'):

        super(HybridEmbedding, self).__init__()
        self.init = initializations.get(init)

        self.unfixed_embed_size = unfixed_embed_size

        self.W_unfixed = self.init((embed_size, embed_dim))
        self.W_fixed = self.init((embed_size, embed_dim))
        self.W_fixed.name = 'HybridEmbedding_fiexed_embed_matrix'

        # print W_fixed
        # for id, row in enumerate(self.W_fixed.get_value()):
        #     if id >= 400: print '[word %d]' % id, row

        self.params = [self.W_unfixed]

        if name is not None:
            self.set_name(name)

    def get_output_mask(self, X):
        return T.ones_like(X) * (1 - T.eq(X, 0))

    def __call__(self, X, mask_zero=False):
        cond = T.lt(X, self.unfixed_embed_size)
        out = T.switch(T.shape_padright(cond), self.W_unfixed[X], self.W_fixed[X])

        if mask_zero:
            return out, self.get_output_mask(X)
        else:
            return out