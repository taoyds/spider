import theano
import theano.tensor as T
import numpy as np
import logging
import copy

from nn.layers.embeddings import Embedding
from nn.layers.core import Dense, Layer
from nn.layers.recurrent import BiLSTM, LSTM, CondAttLSTM
from nn.utils.theano_utils import ndim_itensor, tensor_right_shift, ndim_tensor, alloc_zeros_matrix, shared_zeros
import nn.initializations as initializations
import nn.activations as activations
import nn.optimizers as optimizers

import config
from lang.grammar import Grammar
from parse import *
from astnode import *


class PointerNet(Layer):
    def __init__(self, name='PointerNet'):
        super(PointerNet, self).__init__()

        self.dense1_input = Dense(config.encoder_hidden_dim, config.ptrnet_hidden_dim, activation='linear', name='Dense1_input')

        self.dense1_h = Dense(config.decoder_hidden_dim + config.encoder_hidden_dim, config.ptrnet_hidden_dim, activation='linear', name='Dense1_h')

        self.dense2 = Dense(config.ptrnet_hidden_dim, 1, activation='linear', name='Dense2')

        self.params += self.dense1_input.params + self.dense1_h.params + self.dense2.params

        self.set_name(name)

    def __call__(self, query_embed, query_token_embed_mask, decoder_states):
        query_embed_trans = self.dense1_input(query_embed)
        h_trans = self.dense1_h(decoder_states)

        query_embed_trans = query_embed_trans.dimshuffle((0, 'x', 1, 2))
        h_trans = h_trans.dimshuffle((0, 1, 'x', 2))

        # (batch_size, max_decode_step, query_token_num, ptr_net_hidden_dim)
        dense1_trans = T.tanh(query_embed_trans + h_trans)

        scores = self.dense2(dense1_trans).flatten(3)

        scores = T.exp(scores - T.max(scores, axis=-1, keepdims=True))
        scores *= query_token_embed_mask.dimshuffle((0, 'x', 1))
        scores = scores / T.sum(scores, axis=-1, keepdims=True)

        return scores

class Hyp:
    def __init__(self, *args):
        if isinstance(args[0], Hyp):
            hyp = args[0]
            self.grammar = hyp.grammar
            self.tree = hyp.tree.copy()
            self.t = hyp.t
            self.hist_h = list(hyp.hist_h)
            self.log = hyp.log
            self.has_grammar_error = hyp.has_grammar_error
        else:
            assert isinstance(args[0], Grammar)
            grammar = args[0]
            self.grammar = grammar
            self.tree = DecodeTree(grammar.root_node.type)
            self.t=-1
            self.hist_h = []
            self.log = ''
            self.has_grammar_error = False

        self.score = 0.0

        self.__frontier_nt = self.tree
        self.__frontier_nt_t = -1

    def __repr__(self):
        return self.tree.__repr__()

    def can_expand(self, node):
        if self.grammar.is_value_node(node):
            # if the node is finished
            if node.value is not None and node.value.endswith('<eos>'):
                return False
            return True
        elif self.grammar.is_terminal(node):
            return False

        # elif node.type == 'epsilon':
        #     return False
        # elif is_terminal_ast_type(node.type):
        #     return False

        # if node.type == 'root':
        #     return True
        # elif inspect.isclass(node.type) and issubclass(node.type, ast.AST) and not is_terminal_ast_type(node.type):
        #     return True
        # elif node.holds_value and not node.label.endswith('<eos>'):
        #     return True

        return True

    def apply_rule(self, rule, nt=None):
        if nt is None:
            nt = self.frontier_nt()

        # assert rule.parent.type == nt.type
        if rule.parent.type != nt.type:
            self.has_grammar_error = True

        self.t += 1
        # set the time step when the rule leading by this nt is applied
        nt.t = self.t
        # record the ApplyRule action that is used to expand the current node
        nt.applied_rule = rule

        for child_node in rule.children:
            child = DecodeTree(child_node.type, child_node.label, child_node.value)
            # if is_builtin_type(rule.parent.type):
            #     child.label = None
            #     child.holds_value = True

            nt.add_child(child)

    def append_token(self, token, nt=None):
        if nt is None:
            nt = self.frontier_nt()

        self.t += 1

        if nt.value is None:
            # this terminal node is empty
            nt.t = self.t
            nt.value = token
        else:
            nt.value += token

    def frontier_nt_helper(self, node):
        if node.is_leaf:
            if self.can_expand(node):
                return node
            else:
                return None

        for child in node.children:
            result = self.frontier_nt_helper(child)
            if result:
                return result

        return None

    def frontier_nt(self):
        if self.__frontier_nt_t == self.t:
            return self.__frontier_nt
        else:
            _frontier_nt = self.frontier_nt_helper(self.tree)
            self.__frontier_nt = _frontier_nt
            self.__frontier_nt_t = self.t

            return _frontier_nt

    def get_action_parent_t(self):
        """
        get the time step when the parent of the current
        action was generated
        WARNING: 0 will be returned if parent if None
        """
        nt = self.frontier_nt()

        # if nt is a non-finishing leaf
        # if nt.holds_value:
        #     return nt.t

        if nt.parent:
            return nt.parent.t
        else:
            return 0

    # def get_action_parent_tree(self):
    #     """
    #     get the parent tree
    #     """
    #     nt = self.frontier_nt()
    #
    #     # if nt is a non-finishing leaf
    #     if nt.holds_value:
    #         return nt
    #
    #     if nt.parent:
    #         return nt.parent
    #     else:
    #         return None

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
        self.H_i = self.inner_init((self.output_dim, self.output_dim))
        self.P_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.C_f = self.inner_init((self.context_dim, self.output_dim))
        self.H_f = self.inner_init((self.output_dim, self.output_dim))
        self.P_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.C_c = self.inner_init((self.context_dim, self.output_dim))
        self.H_c = self.inner_init((self.output_dim, self.output_dim))
        self.P_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.C_o = self.inner_init((self.context_dim, self.output_dim))
        self.H_o = self.inner_init((self.output_dim, self.output_dim))
        self.P_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i, self.C_i, self.H_i, self.P_i,
            self.W_c, self.U_c, self.b_c, self.C_c, self.H_c, self.P_c,
            self.W_f, self.U_f, self.b_f, self.C_f, self.H_f, self.P_f,
            self.W_o, self.U_o, self.b_o, self.C_o, self.H_o, self.P_o,
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

        # attention over history
        self.hatt_h_W1 = self.init((output_dim, att_hidden_dim))
        self.hatt_hist_W1 = self.init((output_dim, att_hidden_dim))
        self.hatt_b1 = shared_zeros((att_hidden_dim))

        self.hatt_W2 = self.init((att_hidden_dim, 1))
        self.hatt_b2 = shared_zeros((1))

        self.params += [
            self.hatt_h_W1, self.hatt_hist_W1, self.hatt_b1,
            self.hatt_W2, self.hatt_b2
        ]

        self.set_name(name)

    def _step(self,
              t, xi_t, xf_t, xo_t, xc_t, mask_t, parent_t,
              h_tm1, c_tm1, hist_h,
              u_i, u_f, u_o, u_c,
              c_i, c_f, c_o, c_c,
              h_i, h_f, h_o, h_c,
              p_i, p_f, p_o, p_c,
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
        att_raw = att_raw.reshape((att_raw.shape[0], att_raw.shape[1]))

        # (batch_size, context_size)
        ctx_att = T.exp(att_raw - T.max(att_raw, axis=-1, keepdims=True))

        if context_mask:
            ctx_att = ctx_att * context_mask

        ctx_att = ctx_att / T.sum(ctx_att, axis=-1, keepdims=True)
        # (batch_size, context_dim)
        ctx_vec = T.sum(context * ctx_att[:, :, None], axis=1)

        # t = theano.printing.Print('t')(t)

        ##### attention over history #####

        def _attention_over_history():
            hist_h_mask = T.zeros((hist_h.shape[0], hist_h.shape[1]), dtype='int8')
            hist_h_mask = T.set_subtensor(hist_h_mask[:, T.arange(t)], 1)

            hist_h_att_trans = T.dot(hist_h, self.hatt_hist_W1) + self.hatt_b1
            h_tm1_hatt_trans = T.dot(h_tm1, self.hatt_h_W1)

            hatt_hidden = T.tanh(hist_h_att_trans + h_tm1_hatt_trans[:, None, :])
            hatt_raw = T.dot(hatt_hidden, self.hatt_W2) + self.hatt_b2
            hatt_raw = hatt_raw.reshape((hist_h.shape[0], hist_h.shape[1]))
            # hatt_raw = theano.printing.Print('hatt_raw')(hatt_raw)
            hatt_exp = T.exp(hatt_raw - T.max(hatt_raw, axis=-1, keepdims=True)) * hist_h_mask
            # hatt_exp = theano.printing.Print('hatt_exp')(hatt_exp)
            # hatt_exp = hatt_exp.flatten(2)
            h_att_weights = hatt_exp / (T.sum(hatt_exp, axis=-1, keepdims=True) + 1e-7)
            # h_att_weights = theano.printing.Print('h_att_weights')(h_att_weights)

            # (batch_size, output_dim)
            _h_ctx_vec = T.sum(hist_h * h_att_weights[:, :, None], axis=1)

            return _h_ctx_vec

        h_ctx_vec = T.switch(t,
                             _attention_over_history(),
                             T.zeros_like(h_tm1))

        # h_ctx_vec = theano.printing.Print('h_ctx_vec')(h_ctx_vec)

        ##### attention over history #####

        ##### feed in parent hidden state #####

        if not config.parent_hidden_state_feed:
            t = 0

        par_h = T.switch(t,
                         hist_h[T.arange(hist_h.shape[0]), parent_t, :],
                         T.zeros_like(h_tm1))

        ##### feed in parent hidden state #####
        if config.tree_attention:
            i_t = self.inner_activation(
                xi_t + T.dot(h_tm1 * b_u[0], u_i) + T.dot(ctx_vec, c_i) + T.dot(par_h, p_i) + T.dot(h_ctx_vec, h_i))
            f_t = self.inner_activation(
                xf_t + T.dot(h_tm1 * b_u[1], u_f) + T.dot(ctx_vec, c_f) + T.dot(par_h, p_f) + T.dot(h_ctx_vec, h_f))
            c_t = f_t * c_tm1 + i_t * self.activation(
                xc_t + T.dot(h_tm1 * b_u[2], u_c) + T.dot(ctx_vec, c_c) + T.dot(par_h, p_c) + T.dot(h_ctx_vec, h_c))
            o_t = self.inner_activation(
                xo_t + T.dot(h_tm1 * b_u[3], u_o) + T.dot(ctx_vec, c_o) + T.dot(par_h, p_o) + T.dot(h_ctx_vec, h_o))
        else:
            i_t = self.inner_activation(
                xi_t + T.dot(h_tm1 * b_u[0], u_i) + T.dot(ctx_vec, c_i) + T.dot(par_h, p_i))  # + T.dot(h_ctx_vec, h_i)
            f_t = self.inner_activation(
                xf_t + T.dot(h_tm1 * b_u[1], u_f) + T.dot(ctx_vec, c_f) + T.dot(par_h, p_f))  # + T.dot(h_ctx_vec, h_f)
            c_t = f_t * c_tm1 + i_t * self.activation(
                xc_t + T.dot(h_tm1 * b_u[2], u_c) + T.dot(ctx_vec, c_c) + T.dot(par_h, p_c))  # + T.dot(h_ctx_vec, h_c)
            o_t = self.inner_activation(
                xo_t + T.dot(h_tm1 * b_u[3], u_o) + T.dot(ctx_vec, c_o) + T.dot(par_h, p_o))  # + T.dot(h_ctx_vec, h_o)
        h_t = o_t * self.activation(c_t)

        h_t = (1 - mask_t) * h_tm1 + mask_t * h_t
        c_t = (1 - mask_t) * c_tm1 + mask_t * c_t

        new_hist_h = T.set_subtensor(hist_h[:, t, :], h_t)

        return h_t, c_t, ctx_vec, new_hist_h

    def _for_step(self,
                  xi_t, xf_t, xo_t, xc_t, mask_t,
                  h_tm1, c_tm1,
                  context, context_mask, context_att_trans,
                  hist_h, hist_h_att_trans,
                  b_u):

        # context: (batch_size, context_size, context_dim)

        # (batch_size, att_layer1_dim)
        h_tm1_att_trans = T.dot(h_tm1, self.att_h_W1)

        # (batch_size, context_size, att_layer1_dim)
        att_hidden = T.tanh(context_att_trans + h_tm1_att_trans[:, None, :])

        # (batch_size, context_size, 1)
        att_raw = T.dot(att_hidden, self.att_W2) + self.att_b2

        # (batch_size, context_size)
        ctx_att = T.exp(att_raw).reshape((att_raw.shape[0], att_raw.shape[1]))

        if context_mask:
            ctx_att = ctx_att * context_mask

        ctx_att = ctx_att / T.sum(ctx_att, axis=-1, keepdims=True)

        # (batch_size, context_dim)
        ctx_vec = T.sum(context * ctx_att[:, :, None], axis=1)

        ##### attention over history #####

        if hist_h:
            hist_h = T.stack(hist_h).dimshuffle((1, 0, 2))
            hist_h_att_trans = T.stack(hist_h_att_trans).dimshuffle((1, 0, 2))
            h_tm1_hatt_trans = T.dot(h_tm1, self.hatt_h_W1)

            hatt_hidden = T.tanh(hist_h_att_trans + h_tm1_hatt_trans[:, None, :])
            hatt_raw = T.dot(hatt_hidden, self.hatt_W2) + self.hatt_b2
            hatt_raw = hatt_raw.flatten(2)
            h_att_weights = T.nnet.softmax(hatt_raw)

            # (batch_size, output_dim)
            h_ctx_vec = T.sum(hist_h * h_att_weights[:, :, None], axis=1)
        else:
            h_ctx_vec = T.zeros_like(h_tm1)

        ##### attention over history #####

        i_t = self.inner_activation(xi_t + T.dot(h_tm1 * b_u[0], self.U_i) + T.dot(ctx_vec, self.C_i) + T.dot(h_ctx_vec, self.H_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1 * b_u[1], self.U_f) + T.dot(ctx_vec, self.C_f) + T.dot(h_ctx_vec, self.H_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1 * b_u[2], self.U_c) + T.dot(ctx_vec, self.C_c) + T.dot(h_ctx_vec, self.H_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1 * b_u[3], self.U_o) + T.dot(ctx_vec, self.C_o) + T.dot(h_ctx_vec, self.H_o))
        h_t = o_t * self.activation(c_t)

        h_t = (1 - mask_t) * h_tm1 + mask_t * h_t
        c_t = (1 - mask_t) * c_tm1 + mask_t * c_t

        # ctx_vec = theano.printing.Print('ctx_vec')(ctx_vec)

        return h_t, c_t, ctx_vec

    def __call__(self, X, context, parent_t_seq, init_state=None, init_cell=None, hist_h=None,
                 mask=None, context_mask=None,
                 dropout=0, train=True, srng=None,
                 time_steps=None):
        assert context_mask.dtype == 'int8', 'context_mask is not int8, got %s' % context_mask.dtype

        # (n_timestep, batch_size)
        mask = self.get_mask(mask, X)
        # (n_timestep, batch_size, input_dim)
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

        # (n_timestep, batch_size, output_dim)
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

        if not hist_h:
            # (batch_size, n_timestep, output_dim)
            hist_h = alloc_zeros_matrix(X.shape[1], X.shape[0], self.output_dim)

        if train:
            n_timestep = X.shape[0]
            time_steps = T.arange(n_timestep, dtype='int32')

        # (n_timestep, batch_size)
        parent_t_seq = parent_t_seq.dimshuffle((1, 0))

        [outputs, cells, ctx_vectors, hist_h_outputs], updates = theano.scan(
            self._step,
            sequences=[time_steps, xi, xf, xo, xc, mask, parent_t_seq],
            outputs_info=[
                first_state,  # for h
                first_cell,  # for cell
                None, # T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.context_dim), 1),  # for ctx vector
                hist_h,  # for hist_h
            ],
            non_sequences=[
                self.U_i, self.U_f, self.U_o, self.U_c,
                self.C_i, self.C_f, self.C_o, self.C_c,
                self.H_i, self.H_f, self.H_o, self.H_c,
                self.P_i, self.P_f, self.P_o, self.P_c,
                self.att_h_W1, self.att_W2, self.att_b2,
                context, context_mask, context_att_trans,
                B_u
            ])

        outputs = outputs.dimshuffle((1, 0, 2))
        ctx_vectors = ctx_vectors.dimshuffle((1, 0, 2))
        cells = cells.dimshuffle((1, 0, 2))

        return outputs, cells, ctx_vectors

    def get_mask(self, mask, X):
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)
        mask = mask.astype('int8')

        return mask