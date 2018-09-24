# -*- coding: utf-8 -*-

import logging
import theano
import theano.tensor as T
import numpy as np

from .core import *


class GRU(Layer):
    '''
        Gated Recurrent Unit - Cho et al. 2014

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 return_sequences=False, name='GRU'):

        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if name is not None:
            self.set_name(name)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        # h_tm1 = theano.printing.Print(self.name + 'h_tm1::')(h_tm1)
        h_mask_tm1 = mask_tm1 * h_tm1
        # h_mask_tm1 = theano.printing.Print(self.name + 'h_mask_tm1::')(h_mask_tm1)
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = z * h_mask_tm1 + (1 - z) * hh_t
        return h_t

    def __call__(self, X, mask=None, init_state=None):
        padded_mask = self.get_padded_shuffled_mask(mask, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h

        if init_state:
            # (batch_size, output_dim)
            outputs_info = T.unbroadcast(init_state, 1)
        else:
            outputs_info = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=outputs_info,
            non_sequences=[self.U_z, self.U_r, self.U_h])

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_padded_shuffled_mask(self, mask, X, pad=0):
        # mask is (nb_samples, time)
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8')


class GRU_4BiRNN(Layer):
    '''
        Gated Recurrent Unit - Cho et al. 2014

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 return_sequences=False, name=None):

        super(GRU_4BiRNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if name is not None:
            self.set_name(name)

    def _step(self,
              # xz_t, xr_t, xh_t, mask_tm1, mask,
              xz_t, xr_t, xh_t, mask,
              h_tm1,
              u_z, u_r, u_h):
        # h_mask_tm1 = mask_tm1 * h_tm1
        # h_tm1 = theano.printing.Print(self.name + '::h_tm1::')(h_tm1)
        # mask = theano.printing.Print(self.name + '::mask::')(mask)

        z = self.inner_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * hh_t

        # mask
        h_t = (1 - mask) * h_tm1 + mask * h_t
        # h_t = theano.printing.Print(self.name + '::h_t::')(h_t)

        return h_t

    def __call__(self, X, mask=None, init_state=None):
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)
        mask = mask.astype('int8')
        # mask, padded_mask = self.get_padded_shuffled_mask(mask, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h

        if init_state:
            # (batch_size, output_dim)
            outputs_info = T.unbroadcast(init_state, 1)
        else:
            outputs_info = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        outputs, updates = theano.scan(
            self._step,
            # sequences=[x_z, x_r, x_h, padded_mask, mask],
            sequences=[x_z, x_r, x_h, mask],
            outputs_info=outputs_info,
            non_sequences=[self.U_z, self.U_r, self.U_h])

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_padded_shuffled_mask(self, mask, pad=0):
        assert mask, 'mask cannot be None'
        # mask is (nb_samples, time)
        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            padded_mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8'), padded_mask.astype('int8')


class LSTM(Layer):
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid', return_sequences=False, name='LSTM'):

        super(LSTM, self).__init__()

        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.return_sequences = return_sequences

        self.input_dim = input_dim

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        self.set_name(name)

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_t,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c, b_u):

        i_t = self.inner_activation(xi_t + T.dot(h_tm1 * b_u[0], u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1 * b_u[1], u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1 * b_u[2], u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1 * b_u[3], u_o))
        h_t = o_t * self.activation(c_t)

        h_t = (1 - mask_t) * h_tm1 + mask_t * h_t
        c_t = (1 - mask_t) * c_tm1 + mask_t * c_t

        return h_t, c_t

    def __call__(self, X, mask=None, init_state=None, dropout=0, train=True, srng=None):
        mask = self.get_mask(mask, X)
        X = X.dimshuffle((1, 0, 2))

        retain_prob = 1. - dropout
        B_w = np.ones((4,), dtype=theano.config.floatX)
        B_u = np.ones((4,), dtype=theano.config.floatX)
        if dropout > 0:
            logging.info('applying dropout with p = %f', dropout)
            if train:
                B_w = srng.binomial((4, X.shape[1], self.input_dim), p=retain_prob,
                    dtype=theano.config.floatX)
                B_u = srng.binomial((4, X.shape[1], self.output_dim), p=retain_prob,
                    dtype=theano.config.floatX)
            else:
                B_w *= retain_prob
                B_u *= retain_prob

        xi = T.dot(X * B_w[0], self.W_i) + self.b_i
        xf = T.dot(X * B_w[1], self.W_f) + self.b_f
        xc = T.dot(X * B_w[2], self.W_c) + self.b_c
        xo = T.dot(X * B_w[3], self.W_o) + self.b_o

        if init_state:
            # (batch_size, output_dim)
            first_state = T.unbroadcast(init_state, 1)
        else:
            first_state = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc, mask],
            outputs_info=[
                first_state,
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c, B_u])

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_mask(self, mask, X):
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)
        mask = mask.astype('int8')

        return mask


class BiLSTM(Layer):
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid', return_sequences=False, name='BiLSTM'):
        super(BiLSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_sequences = return_sequences

        params = dict(locals())
        del params['self']

        params['name'] = 'foward_lstm'
        self.forward_lstm = LSTM(**params)
        params['name'] = 'backward_lstm'
        self.backward_lstm = LSTM(**params)

        self.params = self.forward_lstm.params + self.backward_lstm.params

        self.set_name(name)

    def __call__(self, X, mask=None, init_state=None, dropout=0, train=True, srng=None):
        # X: (nb_samples, nb_time_steps, embed_dim)
        # mask: (nb_samples, nb_time_steps)
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        hidden_states_forward = self.forward_lstm(X, mask, init_state, dropout, train, srng)
        hidden_states_backward = self.backward_lstm(X[:, ::-1, :], mask[:, ::-1], init_state, dropout, train, srng)

        if self.return_sequences:
            hidden_states = T.concatenate([hidden_states_forward, hidden_states_backward[:, ::-1, :]], axis=-1)
        else:
            raise NotImplementedError()

        return hidden_states


class CondAttLSTM(Layer):
    """
    Conditional LSTM with Attention
    """
    def __init__(self, input_dim, output_dim,
                 context_dim, att_hidden_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid', name='CondAttLSTM'):

        super(CondAttLSTM, self).__init__()

        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.context_dim = context_dim
        self.input_dim = input_dim

        # regular LSTM layer

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.C_i = self.inner_init((self.context_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.C_f = self.inner_init((self.context_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.C_c = self.inner_init((self.context_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.C_o = self.inner_init((self.context_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i, self.C_i,
            self.W_c, self.U_c, self.b_c, self.C_c,
            self.W_f, self.U_f, self.b_f, self.C_f,
            self.W_o, self.U_o, self.b_o, self.C_o,
        ]

        # attention layer
        self.att_ctx_W1 = self.init((context_dim, att_hidden_dim))
        self.att_h_W1 = self.init((output_dim, att_hidden_dim))
        self.att_b1 = shared_zeros((att_hidden_dim))

        self.att_W2 = self.init((att_hidden_dim, 1))
        self.att_b2 = shared_zeros((1))

        self.params += [
            self.att_ctx_W1, self.att_h_W1, self.att_b1,
            self.att_W2, self.att_b2
        ]

        self.set_name(name)

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_t,
              h_tm1, c_tm1, ctx_vec_tm1,
              u_i, u_f, u_o, u_c, c_i, c_f, c_o, c_c,
              att_h_w1, att_w2, att_b2,
              context, context_mask, context_att_trans,
              b_u):

        # context: (batch_size, context_size, context_dim)

        # (batch_size, att_layer1_dim)
        h_tm1_att_trans = T.dot(h_tm1, att_h_w1)

        # h_tm1_att_trans = theano.printing.Print('h_tm1_att_trans')(h_tm1_att_trans)

        # (batch_size, context_size, att_layer1_dim)
        att_hidden = T.tanh(context_att_trans + h_tm1_att_trans[:, None, :])
        # (batch_size, context_size, 1)
        att_raw = T.dot(att_hidden, att_w2) + att_b2

        # (batch_size, context_size)
        ctx_att = T.exp(att_raw).reshape((att_raw.shape[0], att_raw.shape[1]))

        if context_mask:
            ctx_att = ctx_att * context_mask

        ctx_att = ctx_att / T.sum(ctx_att, axis=-1, keepdims=True)
        # (batch_size, context_dim)
        ctx_vec = T.sum(context * ctx_att[:, :, None], axis=1)

        i_t = self.inner_activation(xi_t + T.dot(h_tm1 * b_u[0], u_i) + T.dot(ctx_vec, c_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1 * b_u[1], u_f) + T.dot(ctx_vec, c_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1 * b_u[2], u_c) + T.dot(ctx_vec, c_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1 * b_u[3], u_o) + T.dot(ctx_vec, c_o))
        h_t = o_t * self.activation(c_t)

        h_t = (1 - mask_t) * h_tm1 + mask_t * h_t
        c_t = (1 - mask_t) * c_tm1 + mask_t * c_t

        # ctx_vec = theano.printing.Print('ctx_vec')(ctx_vec)

        return h_t, c_t, ctx_vec

    def __call__(self, X, context, init_state=None, init_cell=None, mask=None, context_mask=None,
                 dropout=0, train=True, srng=None):
        assert context_mask.dtype == 'int8', 'context_mask is not int8, got %s' % context_mask.dtype

        mask = self.get_mask(mask, X)
        X = X.dimshuffle((1, 0, 2))

        retain_prob = 1. - dropout
        B_w = np.ones((4,), dtype=theano.config.floatX)
        B_u = np.ones((4,), dtype=theano.config.floatX)
        if dropout > 0:
            logging.info('applying dropout with p = %f', dropout)
            if train:
                B_w = srng.binomial((4, X.shape[1], self.input_dim), p=retain_prob,
                                    dtype=theano.config.floatX)
                B_u = srng.binomial((4, X.shape[1], self.output_dim), p=retain_prob,
                                    dtype=theano.config.floatX)
            else:
                B_w *= retain_prob
                B_u *= retain_prob

        xi = T.dot(X * B_w[0], self.W_i) + self.b_i
        xf = T.dot(X * B_w[1], self.W_f) + self.b_f
        xc = T.dot(X * B_w[2], self.W_c) + self.b_c
        xo = T.dot(X * B_w[3], self.W_o) + self.b_o

        # (batch_size, context_size, att_layer1_dim)
        context_att_trans = T.dot(context, self.att_ctx_W1) + self.att_b1

        if init_state:
            # (batch_size, output_dim)
            first_state = T.unbroadcast(init_state, 1)
        else:
            first_state = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        if init_cell:
            # (batch_size, output_dim)
            first_cell = T.unbroadcast(init_cell, 1)
        else:
            first_cell = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        [outputs, cells, ctx_vectors], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc, mask],
            outputs_info=[
                first_state,  # for h
                first_cell,  # for cell   T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.context_dim), 1)  # for ctx vector
            ],
            non_sequences=[
                self.U_i, self.U_f, self.U_o, self.U_c,
                self.C_i, self.C_f, self.C_o, self.C_c,
                self.att_h_W1, self.att_W2, self.att_b2,
                context, context_mask, context_att_trans,
                B_u
            ])

        outputs = outputs.dimshuffle((1, 0, 2))
        ctx_vectors = ctx_vectors.dimshuffle((1, 0, 2))
        cells = cells.dimshuffle((1, 0, 2))

        return outputs, cells, ctx_vectors
        # return outputs[-1]

    def get_mask(self, mask, X):
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)
        mask = mask.astype('int8')

        return mask


class GRUDecoder(Layer):
    '''
        GRU Decoder
    '''
    def __init__(self, input_dim, context_dim, hidden_dim, vocab_num,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 name='GRUDecoder'):

        super(GRUDecoder, self).__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.vocab_num = vocab_num

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.W_z = self.init((self.input_dim, self.hidden_dim))
        self.U_z = self.inner_init((self.hidden_dim, self.hidden_dim))
        self.C_z = self.init((self.context_dim, self.hidden_dim))
        self.b_z = shared_zeros((self.hidden_dim))

        self.W_r = self.init((self.input_dim, self.hidden_dim))
        self.U_r = self.inner_init((self.hidden_dim, self.hidden_dim))
        self.C_r = self.init((self.context_dim, self.hidden_dim))
        self.b_r = shared_zeros((self.hidden_dim))

        self.W_h = self.init((self.input_dim, self.hidden_dim))
        self.U_h = self.inner_init((self.hidden_dim, self.hidden_dim))
        self.C_h = self.init((self.context_dim, self.hidden_dim))
        self.b_h = shared_zeros((self.hidden_dim))

        # self.W_y = self.init((self.input_dim, self.vocab_num))
        self.U_y = self.init((self.hidden_dim, self.vocab_num))
        self.C_y = self.init((self.context_dim, self.vocab_num))
        self.b_y = shared_zeros((self.vocab_num))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
            self.C_z, self.C_r, self.C_h,
            self.U_y, self.C_y, self.b_y, #self.W_y
        ]

        if name is not None:
            self.set_name(name)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = z * h_mask_tm1 + (1 - z) * hh_t
        return h_t

    def __call__(self, target, context, mask=None):
        target = target * T.cast(T.shape_padright(mask), 'float32')
        padded_mask = self.get_padded_shuffled_mask(mask, pad=1)
        # target = theano.printing.Print('X::' + self.name)(target)
        X_shifted = T.concatenate([alloc_zeros_matrix(target.shape[0], 1, self.input_dim), target[:, 0:-1, :]], axis=-2)

        # X = theano.printing.Print('X::' + self.name)(X)
        # X = T.zeros_like(target)
        # T.set_subtensor(X[:, 1:, :], target[:, 0:-1, :])

        X = X_shifted.dimshuffle((1, 0, 2))

        ctx_step = context.dimshuffle(('x', 0, 1))
        x_z = T.dot(X, self.W_z) + T.dot(ctx_step, self.C_z) + self.b_z
        x_r = T.dot(X, self.W_r) + T.dot(ctx_step, self.C_r) + self.b_r
        x_h = T.dot(X, self.W_h) + T.dot(ctx_step, self.C_h) + self.b_h

        h, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h])

        # (batch_size, max_token_len, hidden_dim)
        h = h.dimshuffle((1, 0, 2))

        # (batch_size, max_token_len, vocab_size)
        predicts = T.dot(h, self.U_y) + T.dot(context.dimshuffle((0, 'x', 1)), self.C_y) + self.b_y # + T.dot(X_shifted, self.W_y)

        predicts_flatten = predicts.reshape((-1, predicts.shape[2]))
        return T.nnet.softmax(predicts_flatten).reshape((predicts.shape[0], predicts.shape[1], predicts.shape[2]))

    def get_padded_shuffled_mask(self, mask, pad=0):
        assert mask, 'mask cannot be None'
        # mask is (nb_samples, time)
        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8')
