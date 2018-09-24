import json
import torch
import datetime
import argparse
import numpy as np
from scripts.utils import *
from scripts.model.sqlnet import SQLNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true',
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--dataset', type=str, default='',
            help='to dataset directory where includes train, test and table json file.')
    parser.add_argument('--output', type=str, default='',
            help='output file where predicted SQL queries will be printed on')
    parser.add_argument('--train_emb', action='store_true',
            help='Use trained word embedding for SQLNet.')
    args = parser.parse_args()

    N_word=300
    B_word=42
    if args.toy:
        USE_SMALL=True
        GPU=True
        BATCH_SIZE=15
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=64
    TEST_ENTRY=(True, True, True)  # (AGG, SEL, COND)

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, schemas,\
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(args.dataset, use_small=USE_SMALL)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
            load_used=args.train_emb, use_small=USE_SMALL)

    model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb = args.train_emb)

    print "Loading from sel model..."
    model.sel_pred.load_state_dict(torch.load("saved_models/sel_models.dump"))
    print "Loading from sel model..."
    model.cond_pred.load_state_dict(torch.load("saved_models/cond_models.dump"))
    print "Loading from sel model..."
    model.group_pred.load_state_dict(torch.load("saved_models/group_models.dump"))
    print "Loading from sel model..."
    model.order_pred.load_state_dict(torch.load("saved_models/order_models.dump"))

    print_results(model, BATCH_SIZE, test_sql_data, test_table_data, args.output, schemas, TEST_ENTRY)
