import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net_utils import run_lstm, col_name_encode


class CondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, gpu):
        super(CondPredictor, self).__init__()
        self.N_h = N_h
        self.gpu = gpu

        self.q_lstm = nn.LSTM(input_size=N_word+N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.q_num_att = nn.Linear(N_h, N_h)
        self.col_num_out_q = nn.Linear(N_h, N_h)
        self.col_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 6)) # num of cols: 0-4

        self.q_att = nn.Linear(N_h, N_h)
        self.col_out_q = nn.Linear(N_h, N_h)
        self.col_out_c = nn.Linear(N_h, N_h)
        self.col_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

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

    def forward(self, q_emb_var, q_len, col_emb_var, col_len, x_type_emb_var, gt_cond):
        max_q_len = max(q_len)
        max_col_len = max(col_len)
        B = len(q_len)

        x_emb_concat = torch.cat((q_emb_var, x_type_emb_var), 2)
        q_enc, _ = run_lstm(self.q_lstm, x_emb_concat, q_len)
        col_enc, _ = run_lstm(self.col_lstm, col_emb_var, col_len)

        # Predict column number: 0-4
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
        # get select columns for op prediction
        chosen_col_gt = []
        if gt_cond is None:
            cond_nums = np.argmax(col_num_score.data.cpu().numpy(), axis=1)
            col_scores = col_score.data.cpu().numpy()
            chosen_col_gt = [list(np.argsort(-col_scores[b])[:cond_nums[b]]) for b in range(len(cond_nums))]
        else:
            chosen_col_gt = [[x[0] for x in one_gt_cond] for one_gt_cond in gt_cond]

        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([col_enc[b, x]
                for x in chosen_col_gt[b]] + [col_enc[b, 0]] * (5 - len(chosen_col_gt[b])))
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        # Predict op
        op_att_val = torch.matmul(self.op_att(q_enc).unsqueeze(1),
                col_emb.unsqueeze(3)).squeeze()
        for idx, num in enumerate(q_len):
            if num < max_q_len:
                op_att_val[idx, :, num:] = -100
        op_att = self.softmax(op_att_val.view(-1, max_q_len)).view(B, -1, max_q_len)
        q_weighted_op = (q_enc.unsqueeze(1) * op_att.unsqueeze(3)).sum(2)

        op_score = self.op_out(self.op_out_q(q_weighted_op) +
                            self.op_out_c(col_emb)).squeeze()

        score = (col_num_score, col_score, op_score)

        return score
