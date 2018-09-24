import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net_utils import run_lstm, col_name_encode


class SelPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(SelPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        self.q_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_num_att = nn.Linear(N_h, N_h)
        self.col_num_out_q = nn.Linear(N_h, N_h)
        self.col_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 5)) # num of cols: 1-4

        self.q_att = nn.Linear(N_h, N_h)
        self.col_out_q = nn.Linear(N_h, N_h)
        self.col_out_c = nn.Linear(N_h, N_h)
        self.col_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.agg_num_att = nn.Linear(N_h, N_h)
        self.agg_num_out_q = nn.Linear(N_h, N_h)
        self.agg_num_out_c = nn.Linear(N_h, N_h)
        self.agg_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 5))

        self.agg_att = nn.Linear(N_h, N_h)
        self.agg_out_q = nn.Linear(N_h, N_h)
        self.agg_out_c = nn.Linear(N_h, N_h)
        self.agg_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 6)) #to 5

        self.softmax = nn.Softmax() #dim=1
        self.CE = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax()
        self.mlsml = nn.MultiLabelSoftMarginLoss()
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        if gpu:
            self.cuda()

    def forward(self, q_emb_var, q_len, col_emb_var, col_len, col_num, col_name_len, gt_sel):
        max_q_len = max(q_len)
        max_col_len = max(col_len)
        B = len(q_len)

        q_enc, _ = run_lstm(self.q_lstm, q_emb_var, q_len)
        col_enc, _ = col_name_encode(col_emb_var, col_name_len, col_len, self.col_lstm)
        #col_enc, _ = run_lstm(self.col_lstm, col_emb_var, col_len)

        # Predict column number: 1-3
        # att_val_qc_num: (B, max_col_len, max_q_len)
        att_val_qc_num = torch.bmm(col_enc, self.q_num_att(q_enc).transpose(1, 2))
        for idx, num in enumerate(col_len):
            if num < max_col_len:
                att_val_qc_num[idx, num:, :] = -100
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                att_val_qc_num[idx, :, num:] = -100
        att_prob_qc_num = self.softmax(att_val_qc_num.view((-1, max_q_len))).view(B, -1, max_q_len)
        # q_weighted_num: (B, hid_dim)
        q_weighted_num = (q_enc.unsqueeze(1) * att_prob_qc_num.unsqueeze(3)).sum(2).sum(1)
        # self.col_num_out: (B, 4)
        col_num_score = self.col_num_out(self.col_num_out_q(q_weighted_num))

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

        # get select columns for agg prediction
        chosen_sel_gt = []
        if gt_sel is None:
            sel_nums = [x + 1 for x in list(np.argmax(col_num_score.data.cpu().numpy(), axis=1))]
            sel_col_scores = col_score.data.cpu().numpy()
            chosen_sel_gt = [list(np.argsort(-sel_col_scores[b])[:sel_nums[b]])
                    for b in range(len(sel_nums))]
        else:
            for x in gt_sel:
                curr = x[0]
                curr_sel = [curr]
                for col in x:
                    if col != curr:
                        curr_sel.append(col)
                chosen_sel_gt.append(curr_sel)

        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([col_enc[b, x]
                for x in chosen_sel_gt[b]] + [col_enc[b, 0]] * (5 - len(chosen_sel_gt[b])))
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb) # (B, 4, hd)

        # Predict aggregation
        # q_enc.unsqueeze(1): (B, 1, max_x_len, hd)
        # col_emb.unsqueeze(3): (B, 4, hd, 1)
        # agg_num_att_val.squeeze: (B, 4, max_x_len)
        agg_num_att_val = torch.matmul(self.agg_num_att(q_enc).unsqueeze(1),
                col_emb.unsqueeze(3)).squeeze()
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                agg_num_att_val[idx, :, num:] = -100
        agg_num_att = self.softmax(agg_num_att_val.view(-1, max_q_len)).view(B, -1, max_q_len)
        q_weighted_agg_num = (q_enc.unsqueeze(1) * agg_num_att.unsqueeze(3)).sum(2)
        # (B, 4, 4)
        agg_num_score = self.agg_num_out(self.agg_num_out_q(q_weighted_agg_num) +
                self.agg_num_out_c(col_emb)).squeeze()

        agg_att_val = torch.matmul(self.agg_att(q_enc).unsqueeze(1),
                col_emb.unsqueeze(3)).squeeze()
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                agg_att_val[idx, :, num:] = -100
        agg_att = self.softmax(agg_att_val.view(-1, max_q_len)).view(B, -1, max_q_len)
        q_weighted_agg = (q_enc.unsqueeze(1) * agg_att.unsqueeze(3)).sum(2)

        agg_score = self.agg_out(self.agg_out_q(q_weighted_agg) +
                            self.agg_out_c(col_emb)).squeeze()

        score = (col_num_score, col_score, agg_num_score, agg_score)

        return score
