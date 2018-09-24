import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class WordEmbedding(nn.Module):
    def __init__(self, word_emb, N_word, gpu, SQL_TOK,
            trainable=False):
        super(WordEmbedding, self).__init__()
        self.trainable = trainable
        self.N_word = N_word
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK

        if trainable:
            print "Using trainable embedding"
            self.w2i, word_emb_val = word_emb
            # tranable when using pretrained model, init embedding weights using prev embedding
            self.embedding = nn.Embedding(len(self.w2i), N_word)
            self.embedding.weight = nn.Parameter(torch.from_numpy(word_emb_val.astype(np.float32)))
        else:
            # else use word2vec or glove
            self.word_emb = word_emb
            print "Using fixed embedding for words but trainable embedding for types"


    def gen_xc_type_batch(self, xc_type, is_col=False, is_list=False):
        B = len(xc_type)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_q in enumerate(xc_type):
            if is_list:
                q_val = map(lambda x:self.w2i.get(" ".join(sorted(x)), 0), one_q)
            else:
                q_val = map(lambda x:self.w2i.get(x, 0), one_q)
            if is_col:
                val_embs.append(q_val)  #<BEG> and <END>
                val_len[i] = len(q_val)
            else:
                val_embs.append([1] + q_val + [2])  #<BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
        max_len = max(val_len)
        val_tok_array = np.zeros((B, max_len), dtype=np.int64)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_tok_array[i,t] = val_embs[i][t]
        val_tok = torch.from_numpy(val_tok_array)
        if self.gpu:
            val_tok = val_tok.cuda()
        val_tok_var = Variable(val_tok)
        val_inp_var = self.embedding(val_tok_var)

        return val_inp_var, val_len


    def gen_x_batch(self, q, col, is_list=False, is_q=False):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, (one_q, one_col) in enumerate(zip(q, col)):
            if self.trainable:
                # q_val is only the indexes of words
                q_val = map(lambda x:self.w2i.get(x, 0), one_q)
            elif not is_list:
                q_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), one_q)
            else:
                q_val = []
                for ws in one_q:
                    emb_list = []
                    ws_len = len(ws)
                    for w in ws:
                        emb_list.append(self.word_emb.get(w, np.zeros(self.N_word, dtype=np.float32)))
                    if ws_len == 0:
                        raise Exception("word list should not be empty!")
                    elif ws_len == 1:
                        q_val.append(emb_list[0])
                    else:
                        q_val.append(sum(emb_list) / float(ws_len))

            if self.trainable:
                val_embs.append([1] + q_val + [2])  #<BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
            elif not is_list or is_q:
                val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [np.zeros(self.N_word, dtype=np.float32)])  #<BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
            else:
                val_embs.append(q_val)
                val_len[i] = len(q_val)
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)
        return val_inp_var, val_len


    def gen_col_batch(self, cols):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)
        #TODO: what is the diff bw name_len and col_len?
        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len


    def gen_agg_batch(self, q):
        B = len(q)
        ret = []
        agg_ops = ['none', 'maximum', 'minimum', 'count', 'total', 'average']
        for b in range(B):
            if self.trainable:
                ct_val = map(lambda x:self.w2i.get(x, 0), agg_ops)
            else:
                ct_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), agg_ops)
            ret.append(ct_val)

        agg_emb_array = np.zeros((B, 6, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(ret[i])):
                agg_emb_array[i,t,:] = ret[i][t]
        agg_inp = torch.from_numpy(agg_emb_array)
        if self.gpu:
            agg_inp = agg_inp.cuda()
        agg_inp_var = Variable(agg_inp)

        return agg_inp_var


    def str_list_to_batch(self, str_list):
        """get a list var of wemb of words in each column name in current bactch"""
        B = len(str_list)

        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list):
            if self.trainable:
                val = [self.w2i.get(x, 0) for x in one_str]
            else:
                val = [self.word_emb.get(x, np.zeros(
                    self.N_word, dtype=np.float32)) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros(
                    (B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        return val_inp_var, val_len
