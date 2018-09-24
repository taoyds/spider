import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net_utils import run_lstm, col_name_encode


class GroupPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(GroupPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        self.q_lstm = nn.LSTM(input_size=N_word+N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.gby_num_h = nn.Linear(N_h, N_h)
        self.gby_num_l = nn.Linear(N_h, N_h)
        self.gby_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 4))

        self.gby_att = nn.Linear(N_h, N_h)
        self.gby_out_K = nn.Linear(N_h, N_h)
        self.gby_out_col = nn.Linear(N_h, N_h)
        self.gby_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.hv_att = nn.Linear(N_h, N_h)
        self.hv_out_q = nn.Linear(N_h, N_h)
        self.hv_out_c = nn.Linear(N_h, N_h)
        self.hv_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 2)) #for having/none

        self.q_att = nn.Linear(N_h, N_h)
        self.col_out_q = nn.Linear(N_h, N_h)
        self.col_out_c = nn.Linear(N_h, N_h)
        self.col_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.agg_att = nn.Linear(N_h, N_h)
        self.agg_out_q = nn.Linear(N_h, N_h)
        self.agg_out_c = nn.Linear(N_h, N_h)
        self.agg_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 6)) #to 5

        self.op_att = nn.Linear(N_h, N_h)
        self.op_out_q = nn.Linear(N_h, N_h)
        self.op_out_c = nn.Linear(N_h, N_h)
        self.op_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 12)) #to 5

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if gpu:
            self.cuda()

    def forward(self, q_emb_var, q_len, col_emb_var, col_len, x_type_emb_var):
        max_q_len = max(q_len)
        max_col_len = max(col_len)
        B = len(q_len)

        x_emb_concat = torch.cat((q_emb_var, x_type_emb_var), 2)
        q_enc, _ = run_lstm(self.q_lstm, x_emb_concat, q_len)
        col_enc, _ = run_lstm(self.col_lstm, col_emb_var, col_len)

        # Predict group column number
        gby_num_att = torch.bmm(col_enc, self.gby_num_h(q_enc).transpose(1, 2))
        for idx, num in enumerate(col_len):
            if num < max_col_len:
                gby_num_att[idx, num:, :] = -100
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                gby_num_att[idx, :, num:] = -100

        gby_num_att_val = self.softmax(gby_num_att.view((-1, max_q_len))).view(B, -1, max_q_len)
        gby_num_K = (q_enc.unsqueeze(1) * gby_num_att_val.unsqueeze(3)).sum(2).sum(1)
        gby_num_score = self.gby_num_out(self.gby_num_l(gby_num_K))

        # Predict the group by columns
        gby_att_val = torch.bmm(col_enc, self.gby_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                gby_att_val[idx, :, num:] = -100
        gby_att = self.softmax(gby_att_val.view((-1, max_q_len))).view(B, -1, max_q_len)
        K_gby_expand = (q_enc.unsqueeze(1) * gby_att.unsqueeze(3)).sum(2)
        gby_score = self.gby_out(self.gby_out_K(K_gby_expand) + \
                self.gby_out_col(col_enc)).squeeze()

        for idx, num in enumerate(col_len):
            if num < max_col_len:
                gby_score[idx, num:] = -100

        # Predict Having
        hv_att_val = torch.bmm(col_enc, self.hv_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(col_len):
            if num < max_col_len:
                hv_att_val[idx, num:, :] = -100
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                hv_att_val[idx, :, num:] = -100

        hv_att_prob = self.softmax(hv_att_val.view((-1, max_q_len))).view(B, -1, max_q_len)
        hv_weighted = (q_enc.unsqueeze(1) * hv_att_prob.unsqueeze(3)).sum(2).sum(1)
        hv_score = self.hv_out(self.hv_out_q(hv_weighted))

        # Predict columns.
        att_val_qc = torch.bmm(col_enc, self.q_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc[idx, :, num:] = -100
        att_prob_qc = self.softmax(att_val_qc.view((-1, max_q_len))).view(B, -1, max_q_len)
        # q_weighted: (B, max_col_len, hid_dim)
        q_weighted = (q_enc.unsqueeze(1) * att_prob_qc.unsqueeze(3)).sum(2)
        # Compute prediction scores
        # self.col_out.squeeze(): (B, max_col_len)
        col_score = self.col_out(self.col_out_q(q_weighted) + self.col_out_c(col_enc)).squeeze()
        for idx, num in enumerate(col_len):
            if num < max_col_len:
                col_score[idx, num:] = -100
        # Predict aggregation
        agg_att_val = torch.bmm(col_enc, self.agg_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(col_len):
            if num < max_col_len:
                agg_att_val[idx, num:, :] = -100
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                agg_att_val[idx, :, num:] = -100
        agg_att = self.softmax(agg_att_val.view((-1, max_q_len))).view(B, -1, max_q_len)
        # q_weighted_num: (B, hid_dim)
        q_weighted_agg = (q_enc.unsqueeze(1) * agg_att.unsqueeze(3)).sum(2).sum(1)
        # self.col_num_out: (B, 4)
        agg_score = self.agg_out(self.agg_out_q(q_weighted_agg))


        # Predict op
        op_att_val = torch.matmul(col_enc, self.agg_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                op_att_val[idx, :, num:] = -100
        op_att = self.softmax(op_att_val.view(-1, max_q_len)).view(B, -1, max_q_len)
        q_weighted_op = (q_enc.unsqueeze(1) * op_att.unsqueeze(3)).sum(2).sum(1)

        op_score = self.op_out(self.op_out_q(q_weighted_op))

        score = (gby_num_score, gby_score, hv_score, col_score, agg_score, op_score)

        return score
