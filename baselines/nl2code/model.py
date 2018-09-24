import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

from collections import OrderedDict
import logging
import copy
import heapq
import sys

from nn.layers.embeddings import Embedding
from nn.layers.core import Dense, Dropout, WordDropout
from nn.layers.recurrent import BiLSTM, LSTM
import nn.optimizers as optimizers
import nn.initializations as initializations
from nn.activations import softmax
from nn.utils.theano_utils import *

from config import config_info
import config
from lang.grammar import Grammar
from parse import *
from astnode import *
from util import is_numeric
from components import Hyp, PointerNet, CondAttLSTM

sys.setrecursionlimit(50000)

class Model:
    def __init__(self,vocab,glove_path):
        # self.node_embedding = Embedding(config.node_num, config.node_embed_dim, name='node_embed')

        self.query_embedding = Embedding(config.source_vocab_size, config.word_embed_dim, name='query_embed')
        self.query_embedding.init_pretrained(glove_path,vocab)
        if config.encoder == 'bilstm':
            self.query_encoder_lstm = BiLSTM(config.word_embed_dim, config.encoder_hidden_dim / 2, return_sequences=True,
                                             name='query_encoder_lstm')
        else:
            self.query_encoder_lstm = LSTM(config.word_embed_dim, config.encoder_hidden_dim, return_sequences=True,
                                           name='query_encoder_lstm')

        self.decoder_lstm = CondAttLSTM(config.rule_embed_dim + config.node_embed_dim + config.rule_embed_dim,
                                        config.decoder_hidden_dim, config.encoder_hidden_dim, config.attention_hidden_dim,
                                        name='decoder_lstm')

        self.src_ptr_net = PointerNet()

        self.terminal_gen_softmax = Dense(config.decoder_hidden_dim, 2, activation='softmax', name='terminal_gen_softmax')

        self.rule_embedding_W = initializations.get('normal')((config.rule_num, config.rule_embed_dim), name='rule_embedding_W', scale=0.1)
        self.rule_embedding_b = shared_zeros(config.rule_num, name='rule_embedding_b')

        self.node_embedding = initializations.get('normal')((config.node_num, config.node_embed_dim), name='node_embed', scale=0.1)

        self.vocab_embedding_W = initializations.get('normal')((config.target_vocab_size, config.rule_embed_dim), name='vocab_embedding_W', scale=0.1)
        self.vocab_embedding_b = shared_zeros(config.target_vocab_size, name='vocab_embedding_b')

        # decoder_hidden_dim -> action embed
        self.decoder_hidden_state_W_rule = Dense(config.decoder_hidden_dim, config.rule_embed_dim, name='decoder_hidden_state_W_rule')

        # decoder_hidden_dim -> action embed
        self.decoder_hidden_state_W_token= Dense(config.decoder_hidden_dim + config.encoder_hidden_dim, config.rule_embed_dim,
                                                 name='decoder_hidden_state_W_token')

        # self.rule_encoder_lstm.params
        self.params = self.query_encoder_lstm.params + \
                      self.decoder_lstm.params + self.src_ptr_net.params + self.terminal_gen_softmax.params + \
                      [self.rule_embedding_W, self.rule_embedding_b, self.node_embedding, self.vocab_embedding_W, self.vocab_embedding_b] + \
                      self.decoder_hidden_state_W_rule.params + self.decoder_hidden_state_W_token.params

        self.srng = RandomStreams()

    def build(self):
        # (batch_size, max_example_action_num, action_type)
        tgt_action_seq = ndim_itensor(3, 'tgt_action_seq')

        # (batch_size, max_example_action_num, action_type)
        tgt_action_seq_type = ndim_itensor(3, 'tgt_action_seq_type')

        # (batch_size, max_example_action_num)
        tgt_node_seq = ndim_itensor(2, 'tgt_node_seq')

        # (batch_size, max_example_action_num)
        tgt_par_rule_seq = ndim_itensor(2, 'tgt_par_rule_seq')

        # (batch_size, max_example_action_num)
        tgt_par_t_seq = ndim_itensor(2, 'tgt_par_t_seq')

        # (batch_size, max_example_action_num, symbol_embed_dim)
        # tgt_node_embed = self.node_embedding(tgt_node_seq, mask_zero=False)
        tgt_node_embed = self.node_embedding[tgt_node_seq]

        # (batch_size, max_query_length)
        query_tokens = ndim_itensor(2, 'query_tokens')

        mask = T.TensorType(dtype='int32',name='mask',broadcastable=(True,False))()

        # (batch_size, max_query_length, query_token_embed_dim)
        # (batch_size, max_query_length)
        query_token_embed, query_token_embed_mask = self.query_embedding(query_tokens, mask_zero=True)

        # if WORD_DROPOUT > 0:
        #     logging.info('used word dropout for source, p = %f', WORD_DROPOUT)
        #     query_token_embed, query_token_embed_intact = WordDropout(WORD_DROPOUT, self.srng)(query_token_embed, False)

        batch_size = tgt_action_seq.shape[0]
        max_example_action_num = tgt_action_seq.shape[1]

        # previous action embeddings
        # (batch_size, max_example_action_num, action_embed_dim)
        tgt_action_seq_embed = T.switch(T.shape_padright(tgt_action_seq[:, :, 0] > 0),
                                        self.rule_embedding_W[tgt_action_seq[:, :, 0]],
                                        self.vocab_embedding_W[tgt_action_seq[:, :, 1]])

        tgt_action_seq_embed_tm1 = tensor_right_shift(tgt_action_seq_embed)

        # parent rule application embeddings
        tgt_par_rule_embed = T.switch(tgt_par_rule_seq[:, :, None] < 0,
                                      T.alloc(0., 1, config.rule_embed_dim),
                                      self.rule_embedding_W[tgt_par_rule_seq])

        if not config.frontier_node_type_feed:
            tgt_node_embed *= 0.

        if not config.parent_action_feed:
            tgt_par_rule_embed *= 0.

        # (batch_size, max_example_action_num, action_embed_dim + symbol_embed_dim + action_embed_dim)
        decoder_input = T.concatenate([tgt_action_seq_embed_tm1, tgt_node_embed, tgt_par_rule_embed], axis=-1)

        # (batch_size, max_query_length, query_embed_dim)
        query_embed = self.query_encoder_lstm(query_token_embed, mask=query_token_embed_mask,
                                              dropout=config.dropout, srng=self.srng)

        # (batch_size, max_example_action_num)
        tgt_action_seq_mask = T.any(tgt_action_seq_type, axis=-1)

        # decoder_hidden_states: (batch_size, max_example_action_num, lstm_hidden_state)
        # ctx_vectors: (batch_size, max_example_action_num, encoder_hidden_dim)
        decoder_hidden_states, _, ctx_vectors = self.decoder_lstm(decoder_input,
                                                                  context=query_embed,
                                                                  context_mask=query_token_embed_mask,
                                                                  mask=tgt_action_seq_mask,
                                                                  parent_t_seq=tgt_par_t_seq,
                                                                  dropout=config.dropout,
                                                                  srng=self.srng)

        # if DECODER_DROPOUT > 0:
        #     logging.info('used dropout for decoder output, p = %f', DECODER_DROPOUT)
        #     decoder_hidden_states = Dropout(DECODER_DROPOUT, self.srng)(decoder_hidden_states)

        # ====================================================
        # apply additional non-linearity transformation before
        # predicting actions
        # ====================================================

        decoder_hidden_state_trans_rule = self.decoder_hidden_state_W_rule(decoder_hidden_states)
        decoder_hidden_state_trans_token = self.decoder_hidden_state_W_token(T.concatenate([decoder_hidden_states, ctx_vectors], axis=-1))

        # (batch_size, max_example_action_num, rule_num)
        rule_predict = softmax(T.dot(decoder_hidden_state_trans_rule, T.transpose(self.rule_embedding_W)) + self.rule_embedding_b)

        # (batch_size, max_example_action_num, 2)
        terminal_gen_action_prob = self.terminal_gen_softmax(decoder_hidden_states)

        # (batch_size, max_example_action_num, target_vocab_size)
        logits = T.dot(decoder_hidden_state_trans_token, T.transpose(self.vocab_embedding_W)) + self.vocab_embedding_b
        # vocab_predict = softmax(T.dot(decoder_hidden_state_trans_token, T.transpose(self.vocab_embedding_W)) + self.vocab_embedding_b)
        vocab_predict = softmax(logits.transpose(1,0,2)*mask + (T.min(logits.transpose(1,0,2),axis=1,keepdims=True)-1)*(1-mask)).transpose(1,0,2)
        # (batch_size, max_example_action_num, lstm_hidden_state + encoder_hidden_dim)
        ptr_net_decoder_state = T.concatenate([decoder_hidden_states, ctx_vectors], axis=-1)

        # (batch_size, max_example_action_num, max_query_length)
        copy_prob = self.src_ptr_net(query_embed, query_token_embed_mask, ptr_net_decoder_state)

        # (batch_size, max_example_action_num)
        rule_tgt_prob = rule_predict[T.shape_padright(T.arange(batch_size)),
                                     T.shape_padleft(T.arange(max_example_action_num)),
                                     tgt_action_seq[:, :, 0]]

        # (batch_size, max_example_action_num)
        vocab_tgt_prob = vocab_predict[T.shape_padright(T.arange(batch_size)),
                                       T.shape_padleft(T.arange(max_example_action_num)),
                                       tgt_action_seq[:, :, 1]]

        # (batch_size, max_example_action_num)
        copy_tgt_prob = copy_prob[T.shape_padright(T.arange(batch_size)),
                                  T.shape_padleft(T.arange(max_example_action_num)),
                                  tgt_action_seq[:, :, 2]]


        # (batch_size, max_example_action_num)
        tgt_prob = tgt_action_seq_type[:, :, 0] * rule_tgt_prob + \
                   tgt_action_seq_type[:, :, 1] * terminal_gen_action_prob[:, :, 0] * vocab_tgt_prob + \
                   tgt_action_seq_type[:, :, 2] * terminal_gen_action_prob[:, :, 1] * copy_tgt_prob

        likelihood = T.log(tgt_prob + 1.e-7 * (1 - tgt_action_seq_mask))
        loss = - (likelihood * tgt_action_seq_mask).sum(axis=-1) # / tgt_action_seq_mask.sum(axis=-1)
        loss = T.mean(loss)

        # let's build the function!
        train_inputs = [query_tokens, tgt_action_seq, tgt_action_seq_type,
                        tgt_node_seq, tgt_par_rule_seq, tgt_par_t_seq,mask]
        optimizer = optimizers.get(config.optimizer)
        optimizer.clip_grad = config.clip_grad
        updates, grads = optimizer.get_updates(self.params, loss)
        self.train_func = theano.function(train_inputs, [loss],
                                          # [loss, tgt_action_seq_type, tgt_action_seq,
                                          #  rule_tgt_prob, vocab_tgt_prob, copy_tgt_prob,
                                          #  copy_prob, terminal_gen_action_prob],
                                          updates=updates)

        # if WORD_DROPOUT > 0:
        #     self.build_decoder(query_tokens, query_token_embed_intact, query_token_embed_mask)
        # else:
        #     self.build_decoder(query_tokens, query_token_embed, query_token_embed_mask)

        self.build_decoder(query_tokens, query_token_embed, query_token_embed_mask,mask)

    def build_decoder(self, query_tokens, query_token_embed, query_token_embed_mask,mask):
        logging.info('building decoder ...')

        # mask = ndim_itensor(2, 'mask')

        # (batch_size, decoder_state_dim)
        decoder_prev_state = ndim_tensor(2, name='decoder_prev_state')

        # (batch_size, decoder_state_dim)
        decoder_prev_cell = ndim_tensor(2, name='decoder_prev_cell')

        # (batch_size, n_timestep, decoder_state_dim)
        hist_h = ndim_tensor(3, name='hist_h')

        # (batch_size, decoder_state_dim)
        prev_action_embed = ndim_tensor(2, name='prev_action_embed')

        # (batch_size)
        node_id = T.ivector(name='node_id')

        # (batch_size, node_embed_dim)
        node_embed = self.node_embedding[node_id]

        # (batch_size)
        par_rule_id = T.ivector(name='par_rule_id')

        # (batch_size, decoder_state_dim)
        par_rule_embed = T.switch(par_rule_id[:, None] < 0,
                                  T.alloc(0., 1, config.rule_embed_dim),
                                  self.rule_embedding_W[par_rule_id])

        # ([time_step])
        time_steps = T.ivector(name='time_steps')

        # (batch_size)
        parent_t = T.ivector(name='parent_t')

        # (batch_size, 1)
        parent_t_reshaped = T.shape_padright(parent_t)

        # mask = ndim_itensor(2, 'mask')

        query_embed = self.query_encoder_lstm(query_token_embed, mask=query_token_embed_mask,
                                              dropout=config.dropout, train=False)

        # (batch_size, 1, decoder_state_dim)
        prev_action_embed_reshaped = prev_action_embed.dimshuffle((0, 'x', 1))

        # (batch_size, 1, node_embed_dim)
        node_embed_reshaped = node_embed.dimshuffle((0, 'x', 1))

        # (batch_size, 1, node_embed_dim)
        par_rule_embed_reshaped = par_rule_embed.dimshuffle((0, 'x', 1))

        if not config.frontier_node_type_feed:
            node_embed_reshaped *= 0.

        if not config.parent_action_feed:
            par_rule_embed_reshaped *= 0.

        decoder_input = T.concatenate([prev_action_embed_reshaped, node_embed_reshaped, par_rule_embed_reshaped], axis=-1)

        # (batch_size, 1, decoder_state_dim)
        # (batch_size, 1, decoder_state_dim)
        # (batch_size, 1, field_token_encode_dim)
        decoder_next_state_dim3, decoder_next_cell_dim3, ctx_vectors = self.decoder_lstm(decoder_input,
                                                                                         init_state=decoder_prev_state,
                                                                                         init_cell=decoder_prev_cell,
                                                                                         hist_h=hist_h,
                                                                                         context=query_embed,
                                                                                         context_mask=query_token_embed_mask,
                                                                                         parent_t_seq=parent_t_reshaped,
                                                                                         dropout=config.dropout,
                                                                                         train=False,
                                                                                         time_steps=time_steps)

        decoder_next_state = decoder_next_state_dim3.flatten(2)
        # decoder_output = decoder_next_state * (1 - DECODER_DROPOUT)

        decoder_next_cell = decoder_next_cell_dim3.flatten(2)

        decoder_next_state_trans_rule = self.decoder_hidden_state_W_rule(decoder_next_state)
        decoder_next_state_trans_token = self.decoder_hidden_state_W_token(T.concatenate([decoder_next_state, ctx_vectors.flatten(2)], axis=-1))

        rule_prob = softmax(T.dot(decoder_next_state_trans_rule, T.transpose(self.rule_embedding_W)) + self.rule_embedding_b)

        gen_action_prob = self.terminal_gen_softmax(decoder_next_state)

        # vocab_prob = softmax(T.dot(decoder_next_state_trans_token, T.transpose(self.vocab_embedding_W)) + self.vocab_embedding_b)
        logits = T.dot(decoder_next_state_trans_token, T.transpose(self.vocab_embedding_W)) + self.vocab_embedding_b
        # vocab_predict = softmax(T.dot(decoder_hidden_state_trans_token, T.transpose(self.vocab_embedding_W)) + self.vocab_embedding_b)
        test = T.dot((T.min(logits,axis=1,keepdims=True) - 1) , (1 - mask).reshape((1,mask.shape[1])))
        vocab_prob = softmax(logits*mask + test)
        # vocab_prob = softmax(
        #     logits.transpose(1, 0, 2) * mask + (T.min(logits.transpose(1, 0, 2), axis=1, keepdims=True) - 1) * (
        #     1 - mask)).transpose(1, 0, 2)

        ptr_net_decoder_state = T.concatenate([decoder_next_state_dim3, ctx_vectors], axis=-1)

        copy_prob = self.src_ptr_net(query_embed, query_token_embed_mask, ptr_net_decoder_state)

        copy_prob = copy_prob.flatten(2)

        inputs = [query_tokens]
        outputs = [query_embed, query_token_embed_mask]

        self.decoder_func_init = theano.function(inputs, outputs)

        inputs = [time_steps, decoder_prev_state, decoder_prev_cell, hist_h, prev_action_embed,
                  node_id, par_rule_id, parent_t,
                  query_embed, query_token_embed_mask,mask]

        outputs = [decoder_next_state, decoder_next_cell,
                   rule_prob, gen_action_prob, vocab_prob, copy_prob]

        self.decoder_func_next_step = theano.function(inputs, outputs)

    def decode(self, example, grammar, terminal_vocab, beam_size, max_time_step, log=False):
        # beam search decoding

        eos = 1
        unk = terminal_vocab.unk
        vocab_embedding = self.vocab_embedding_W.get_value(borrow=True)
        rule_embedding = self.rule_embedding_W.get_value(borrow=True)

        query_tokens = example.data[0]
        mask = example.mask.reshape((1,example.mask.shape[0]))

        query_embed, query_token_embed_mask = self.decoder_func_init(query_tokens)

        completed_hyps = []
        completed_hyp_num = 0
        live_hyp_num = 1

        root_hyp = Hyp(grammar)
        root_hyp.state = np.zeros(config.decoder_hidden_dim).astype('float32')
        root_hyp.cell = np.zeros(config.decoder_hidden_dim).astype('float32')
        root_hyp.action_embed = np.zeros(config.rule_embed_dim).astype('float32')
        root_hyp.node_id = grammar.get_node_type_id(root_hyp.tree.type)
        root_hyp.parent_rule_id = -1

        hyp_samples = [root_hyp]  # [list() for i in range(live_hyp_num)]

        # source word id in the terminal vocab
        src_token_id = [terminal_vocab[t] for t in example.query][:config.max_query_length]
        unk_pos_list = [x for x, t in enumerate(src_token_id) if t == unk]

        # sometimes a word may appear multi-times in the source, in this case,
        # we just copy its first appearing position. Therefore we mask the words
        # appearing second and onwards to -1
        token_set = set()
        for i, tid in enumerate(src_token_id):
            if tid in token_set:
                src_token_id[i] = -1
            else: token_set.add(tid)

        for t in xrange(max_time_step):
            hyp_num = len(hyp_samples)
            # print 'time step [%d]' % t
            decoder_prev_state = np.array([hyp.state for hyp in hyp_samples]).astype('float32')
            decoder_prev_cell = np.array([hyp.cell for hyp in hyp_samples]).astype('float32')

            hist_h = np.zeros((hyp_num, max_time_step, config.decoder_hidden_dim)).astype('float32')

            if t > 0:
                for i, hyp in enumerate(hyp_samples):
                    hist_h[i, :len(hyp.hist_h), :] = hyp.hist_h
                    # for j, h in enumerate(hyp.hist_h):
                    #    hist_h[i, j] = h

            prev_action_embed = np.array([hyp.action_embed for hyp in hyp_samples]).astype('float32')
            node_id = np.array([hyp.node_id for hyp in hyp_samples], dtype='int32')
            parent_rule_id = np.array([hyp.parent_rule_id for hyp in hyp_samples], dtype='int32')
            parent_t = np.array([hyp.get_action_parent_t() for hyp in hyp_samples], dtype='int32')
            query_embed_tiled = np.tile(query_embed, [live_hyp_num, 1, 1])
            query_token_embed_mask_tiled = np.tile(query_token_embed_mask, [live_hyp_num, 1])

            inputs = [np.array([t], dtype='int32'), decoder_prev_state, decoder_prev_cell, hist_h, prev_action_embed,
                      node_id, parent_rule_id, parent_t,
                      query_embed_tiled, query_token_embed_mask_tiled,mask]

            decoder_next_state, decoder_next_cell, \
            rule_prob, gen_action_prob, vocab_prob, copy_prob  = self.decoder_func_next_step(*inputs)

            new_hyp_samples = []

            cut_off_k = beam_size
            score_heap = []

            # iterating over items in the beam
            # print 'time step: %d, hyp num: %d' % (t, live_hyp_num)

            word_prob = gen_action_prob[:, 0:1] * vocab_prob
            word_prob[:, unk] = 0

            hyp_scores = np.array([hyp.score for hyp in hyp_samples])

            # word_prob[:, src_token_id] += gen_action_prob[:, 1:2] * copy_prob[:, :len(src_token_id)]
            # word_prob[:, unk] = 0

            rule_apply_cand_hyp_ids = []
            rule_apply_cand_scores = []
            rule_apply_cand_rules = []
            rule_apply_cand_rule_ids = []

            hyp_frontier_nts = []
            word_gen_hyp_ids = []
            cand_copy_probs = []
            unk_words = []

            for k in xrange(live_hyp_num):
                hyp = hyp_samples[k]

                # if k == 0:
                #     print 'Top Hyp: %s' % hyp.tree.__repr__()

                frontier_nt = hyp.frontier_nt()
                hyp_frontier_nts.append(frontier_nt)

                assert hyp, 'none hyp!'

                # if it's not a leaf
                if not grammar.is_value_node(frontier_nt):
                    # iterate over all the possible rules
                    rules = grammar[frontier_nt.as_type_node] if config.head_nt_constraint else grammar
                    assert len(rules) > 0, 'fail to expand nt node %s' % frontier_nt
                    for rule in rules:
                        rule_id = grammar.rule_to_id[rule]

                        cur_rule_score = np.log(rule_prob[k, rule_id])
                        new_hyp_score = hyp.score + cur_rule_score

                        rule_apply_cand_hyp_ids.append(k)
                        rule_apply_cand_scores.append(new_hyp_score)
                        rule_apply_cand_rules.append(rule)
                        rule_apply_cand_rule_ids.append(rule_id)

                else:  # it's a leaf that holds values
                    cand_copy_prob = 0.0
                    for i, tid in enumerate(src_token_id):
                        if tid != -1:
                            word_prob[k, tid] += gen_action_prob[k, 1] * copy_prob[k, i]
                            cand_copy_prob = gen_action_prob[k, 1]

                    # and unk copy probability
                    if len(unk_pos_list) > 0:
                        unk_pos = copy_prob[k, unk_pos_list].argmax()
                        unk_pos = unk_pos_list[unk_pos]

                        unk_copy_score = gen_action_prob[k, 1] * copy_prob[k, unk_pos]
                        word_prob[k, unk] = unk_copy_score

                        unk_word = example.query[unk_pos]
                        unk_words.append(unk_word)

                        cand_copy_prob = gen_action_prob[k, 1]

                    word_gen_hyp_ids.append(k)
                    cand_copy_probs.append(cand_copy_prob)

            # prune the hyp space
            if completed_hyp_num >= beam_size:
                break

            word_prob = np.log(word_prob)

            word_gen_hyp_num = len(word_gen_hyp_ids)
            rule_apply_cand_num = len(rule_apply_cand_scores)

            if word_gen_hyp_num > 0:
                word_gen_cand_scores = hyp_scores[word_gen_hyp_ids, None] + word_prob[word_gen_hyp_ids, :]
                word_gen_cand_scores_flat = word_gen_cand_scores.flatten()

                cand_scores = np.concatenate([rule_apply_cand_scores, word_gen_cand_scores_flat])
            else:
                cand_scores = np.array(rule_apply_cand_scores)

            top_cand_ids = (-cand_scores).argsort()[:beam_size - completed_hyp_num]

            # expand_cand_num = 0
            for cand_id in top_cand_ids:
                # cand is rule application
                new_hyp = None
                if cand_id < rule_apply_cand_num:
                    hyp_id = rule_apply_cand_hyp_ids[cand_id]
                    hyp = hyp_samples[hyp_id]
                    rule_id = rule_apply_cand_rule_ids[cand_id]
                    rule = rule_apply_cand_rules[cand_id]
                    new_hyp_score = rule_apply_cand_scores[cand_id]

                    new_hyp = Hyp(hyp)
                    new_hyp.apply_rule(rule)

                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.hist_h.append(copy.copy(new_hyp.state))
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = rule_embedding[rule_id]
                else:
                    tid = (cand_id - rule_apply_cand_num) % word_prob.shape[1]
                    word_gen_hyp_id = (cand_id - rule_apply_cand_num) / word_prob.shape[1]
                    hyp_id = word_gen_hyp_ids[word_gen_hyp_id]

                    if tid == unk:
                        token = unk_words[word_gen_hyp_id]
                    else:
                        token = terminal_vocab.id_token_map[tid]

                    frontier_nt = hyp_frontier_nts[hyp_id]
                    # if frontier_nt.type == int and (not (is_numeric(token) or token == '<eos>')):
                    #     continue

                    hyp = hyp_samples[hyp_id]
                    new_hyp_score = word_gen_cand_scores[word_gen_hyp_id, tid]

                    new_hyp = Hyp(hyp)
                    new_hyp.append_token(token)

                    if log:
                        cand_copy_prob = cand_copy_probs[word_gen_hyp_id]
                        if cand_copy_prob > 0.5:
                            new_hyp.log += ' || ' + str(new_hyp.frontier_nt()) + '{copy[%s][p=%f]}' % (token ,cand_copy_prob)

                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.hist_h.append(copy.copy(new_hyp.state))
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = vocab_embedding[tid]
                    new_hyp.node_id = grammar.get_node_type_id(frontier_nt)


                # get the new frontier nt after rule application
                new_frontier_nt = new_hyp.frontier_nt()

                # if new_frontier_nt is None, then we have a new completed hyp!
                if new_frontier_nt is None:
                    # if t <= 1:
                    #     continue

                    new_hyp.n_timestep = t + 1
                    completed_hyps.append(new_hyp)
                    completed_hyp_num += 1

                else:
                    new_hyp.node_id = grammar.get_node_type_id(new_frontier_nt.type)
                    # new_hyp.parent_rule_id = grammar.rule_to_id[
                    #     new_frontier_nt.parent.to_rule(include_value=False)]
                    new_hyp.parent_rule_id = grammar.rule_to_id[new_frontier_nt.parent.applied_rule]

                    new_hyp_samples.append(new_hyp)

                # expand_cand_num += 1
                # if expand_cand_num >= beam_size - completed_hyp_num:
                #     break

                # cand is word generation

            live_hyp_num = min(len(new_hyp_samples), beam_size - completed_hyp_num)
            if live_hyp_num < 1:
                break

            hyp_samples = new_hyp_samples
            # hyp_samples = sorted(new_hyp_samples, key=lambda x: x.score, reverse=True)[:live_hyp_num]

        completed_hyps = sorted(completed_hyps, key=lambda x: x.score, reverse=True)

        return completed_hyps

    @property
    def params_name_to_id(self):
        name_to_id = dict()
        for i, p in enumerate(self.params):
            assert p.name is not None
            # print 'parameter [%s]' % p.name

            name_to_id[p.name] = i

        return name_to_id

    @property
    def params_dict(self):
        assert len(set(p.name for p in self.params)) == len(self.params), 'param name clashes!'
        return OrderedDict((p.name, p) for p in self.params)

    def pull_params(self):
        return OrderedDict([(p_name, p.get_value(borrow=False)) for (p_name, p) in self.params_dict.iteritems()])

    def save(self, model_file, **kwargs):
        logging.info('save model to [%s]', model_file)

        weights_dict = self.pull_params()
        for k, v in kwargs.iteritems():
            weights_dict[k] = v

        np.savez(model_file, **weights_dict)

    def load(self, model_file):
        logging.info('load model from [%s]', model_file)
        weights_dict = np.load(model_file)

        # assert len(weights_dict.files) == len(self.params_dict)

        for p_name, p in self.params_dict.iteritems():
            if p_name not in weights_dict:
                raise RuntimeError('parameter [%s] not in saved weights file', p_name)
            else:
                logging.info('loading parameter [%s]', p_name)
                assert np.array_equal(p.shape.eval(), weights_dict[p_name].shape), \
                    'shape mis-match for [%s]!, %s != %s' % (p_name, p.shape.eval(), weights_dict[p_name].shape)

                p.set_value(weights_dict[p_name])
