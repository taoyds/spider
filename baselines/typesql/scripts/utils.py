import re
import io
import json
import numpy as np
import os
#from lib.dbengine import DBEngine

def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.iteritems())
    else:
        return x

def get_main_table_name(file_path):
    prefix_pattern = re.compile('(processed/.*/)(.*)(_table\.json)')
    if prefix_pattern.search(file_path):
        return prefix_pattern.search(file_path).group(2)
    return None

def load_data_new(sql_path, table_data, use_small=False):
    sql_data = []

    print "Loading data from %s"%sql_path
    with open(sql_path) as inf:
        data = lower_keys(json.load(inf))
        sql_data += data

    sql_data_new, table_data_new = process(sql_data, table_data)  # comment out if not on full dataset

    schemas = {}
    for tab in table_data:
        schemas[tab['db_id']] = tab

    if use_small:
        return sql_data_new[:80], table_data_new, schemas
    else:
        return sql_data_new, table_data_new, schemas


def load_dataset(dataset_dir, use_small=False):
    print "Loading from datasets..."
    
    TABLE_PATH = os.path.join(dataset_dir, "tables.json")
    TRAIN_PATH = os.path.join(dataset_dir, "train_type.json")
    DEV_PATH = os.path.join(dataset_dir, "dev_type.json")
    TEST_PATH = os.path.join(dataset_dir, "dev_type.json")
    with open(TABLE_PATH) as inf:
        print "Loading data from %s"%TABLE_PATH
        table_data= json.load(inf)
    train_sql_data, train_table_data, schemas_all = load_data_new(TRAIN_PATH, table_data, use_small=use_small)
    val_sql_data, val_table_data, schemas = load_data_new(DEV_PATH, table_data, use_small=use_small)
    test_sql_data, test_table_data, schemas = load_data_new(TEST_PATH, table_data, use_small=use_small)

    TRAIN_DB = '../alt/data/train.db'
    DEV_DB = '../alt/data/dev.db'
    TEST_DB = '../alt/data/test.db'

    return train_sql_data, train_table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data, schemas_all, TRAIN_DB, DEV_DB, TEST_DB


def to_batch_seq(sql_data, table_data, idxes, st, ed, schemas, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    query_seq = []
    gt_cond_seq = []
    vis_seq = []
    q_type = []
    col_org_seq = []
    schema_seq = []

    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        col_org_seq.append(sql['col_org'])
        q_seq.append(sql['question_tok_concol'])
        q_type.append(sql["question_type_concol_list"])
        table = table_data[sql['table_id']]
        schema_seq.append(schemas[sql['table_id']])
        col_num.append(len(table['col_map']))
        tab_cols = [col[1] for col in table['col_map']]
        col_seq.append([x.split(" ") for x in tab_cols])
        ans_seq.append((sql['agg'],     # sel agg # 0
            sql['sel'],                 # sel col # 1
            len(sql['cond']),           # cond # 2
            tuple(x[0] for x in sql['cond']), # cond col 3
            tuple(x[1] for x in sql['cond']), # cond op 4
            len(set(sql['sel'])),       # number of unique select cols 5
            sql['group'][:-1],          # group by columns 6
            len(sql['group']) - 1,      # number of group by columns 7
            sql['order'][0],            # order by aggregations 8
            sql['order'][1],            # order by columns 9
            len(sql['order'][1]),       # num order by columns 10
            sql['order'][2],            # order by parity 11
            sql['group'][-1][0],        # having agg 12
            sql['group'][-1][1],        # having col 13
            sql['group'][-1][2]         # having op 14
            ))
        #order: [[agg], [col], [dat]]
        #group: [col1, col2, [agg, col, op]]
        query_seq.append(sql['query_tok'])
        gt_cond_seq.append([x for x in sql['cond']])
        vis_seq.append((sql['question'], tab_cols, sql['query']))

    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, vis_seq, col_org_seq, schema_seq, q_type
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, col_org_seq, schema_seq, q_type


def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        query_gt.append(sql_data[idxes[i]])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids


def epoch_train(model, optimizer, batch_size, sql_data, table_data, schemas, pred_entry):
    model.train()
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, col_org_seq, schema_seq, q_type = \
                to_batch_seq(sql_data, table_data, perm, st, ed, schemas)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, q_type, pred_entry, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
        loss = model.loss(score, ans_seq, pred_entry)
        cum_loss += loss.data.cpu().numpy()[0]*(ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        st = ed

    return cum_loss / len(sql_data)


def epoch_acc(model, batch_size, sql_data, table_data, schemas, pred_entry, error_print=False, train_flag = False):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq,\
         raw_data, col_org_seq, schema_seq, q_type = to_batch_seq(sql_data, table_data, perm, st, ed, schemas, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        score = model.forward(q_seq, col_seq, col_num, q_type, pred_entry)
        pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq, raw_col_seq, pred_entry)
        one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt, pred_entry, error_print)
        gen_sqls = model.gen_sql(score, col_org_seq, schema_seq)
        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        st = ed
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)


def print_results(model, batch_size, sql_data, table_data, output_file, schemas, pred_entry, error_print=False, train_flag = False):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    output =  open(output_file, 'w')
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq,\
         raw_data, col_org_seq, schema_seq, q_type = to_batch_seq(sql_data, table_data, perm, st, ed, schemas, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        score = model.forward(q_seq, col_seq, col_num, q_type, pred_entry)
        gen_sqls = model.gen_sql(score, col_org_seq, schema_seq)
        for sql in gen_sqls:
            output.write(sql+"\n")
        st = ed


def load_para_wemb(file_name):
    f = io.open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    ret = {}
    if len(lines[0].split()) == 2:
        lines.pop(0)
    for (n,line) in enumerate(lines):
        info = line.strip().split(' ')
        if info[0].lower() not in ret:
            ret[info[0]] = np.array(map(lambda x:float(x), info[1:]))

    return ret


def load_comb_wemb(fn1, fn2):
    wemb1 = load_word_emb(fn1)
    wemb2 = load_para_wemb(fn2)
    comb_emb = {k: wemb1.get(k, 0) + wemb2.get(k, 0) for k in set(wemb1) | set(wemb2)}

    return comb_emb


def load_concat_wemb(fn1, fn2):
    wemb1 = load_word_emb(fn1)
    wemb2 = load_para_wemb(fn2)
    backup = np.zeros(300, dtype=np.float32)
    comb_emb = {k: np.concatenate((wemb1.get(k, backup), wemb2.get(k, backup)), axis=0) for k in set(wemb1) | set(wemb2)}

    return comb_emb


def load_word_emb(file_name, load_used=False, use_small=False):
    if not load_used:
        print ('Loading word embedding from %s'%file_name)
        ret = {}
        with open(file_name) as inf:
            for idx, line in enumerate(inf):
                if (use_small and idx >= 5000):
                    break
                info = line.strip().split(' ')
                if info[0].lower() not in ret:
                    ret[info[0]] = np.array(map(lambda x:float(x), info[1:]))
        return ret
    else:
        print ('Load used word embedding')
        with open('../alt/glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('../alt/glove/usedwordemb.npy') as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val


def process(sql_data, table_data):
    output_tab = {}
    tables = {}
    for i in range(len(table_data)):
        table = table_data[i]
        temp = {}
        temp['col_map'] = table['column_names']
        db_name = table['db_id']
        # print table
        output_tab[db_name] = temp
        tables[db_name] = table

    output_sql = []
    for i in range(len(sql_data)):
        sql = sql_data[i]
        sql_temp = {}

        # add query metadata
        sql_temp['question'] = sql['question']
        sql_temp["question_type_concol_list"] = sql["question_type_concol_list"]
        sql_temp['question_tok_concol'] = sql['question_tok_concol']
        sql_temp['query'] = sql['query']
        sql_temp['query_tok'] = sql['query_toks']
        sql_temp['table_id'] = sql['db_id']
        table = tables[sql['db_id']]
        sql_temp['col_org'] = table['column_names_original']
        sql_temp['table_org'] = table['table_names_original']
        sql_temp['fk_info'] = table['foreign_keys']
        # process agg/sel
        sql_temp['agg'] = []
        sql_temp['sel'] = []
        gt_sel = sql['sql']['select'][1]
        if len(gt_sel) > 3:
            gt_sel = gt_sel[:3]
        for tup in gt_sel:
            sql_temp['agg'].append(tup[0])
            sql_temp['sel'].append(tup[1][1][1]) #GOLD for sel and agg

        # process where conditions and conjuctions
        sql_temp['cond'] = []
        gt_cond = sql['sql']['where']
        if len(gt_cond) > 0:
            conds = [gt_cond[x] for x in range(len(gt_cond)) if x % 2 == 0]
            for cond in conds:
                curr_cond = []
                curr_cond.append(cond[2][1][1])
                curr_cond.append(cond[1])
                if cond[4] is not None:
                    curr_cond.append([cond[3], cond[4]])
                else:
                    curr_cond.append(cond[3])
                sql_temp['cond'].append(curr_cond) #GOLD for COND [[col, op],[]]

        sql_temp['conj'] = [gt_cond[x] for x in range(len(gt_cond)) if x % 2 == 1]

        # process group by / having
        sql_temp['group'] = [x[1] for x in sql['sql']['groupby']] #assume only one groupby
        having_cond = []
        if len(sql['sql']['having']) > 0:
            gt_having = sql['sql']['having'][0] # currently only do first having condition
            having_cond.append([gt_having[2][1][0]]) # aggregator
            having_cond.append([gt_having[2][1][1]]) # column
            having_cond.append([gt_having[1]]) # operator
            if gt_having[4] is not None:
                having_cond.append([gt_having[3], gt_having[4]])
            else:
                having_cond.append(gt_having[3])
        else:
            having_cond = [[], [], []]
        sql_temp['group'].append(having_cond) #GOLD for GROUP [[col1, col2, [agg, col, op]], [col, []]]

        # process order by / limit
        order_aggs = []
        order_cols = []
        sql_temp['order'] = []
        order_par = 4
        gt_order = sql['sql']['orderby']
        limit = sql['sql']['limit']
        if len(gt_order) > 0:
            order_aggs = [x[1][0] for x in gt_order[1][:1]] # limit to 1 order by
            order_cols = [x[1][1] for x in gt_order[1][:1]]
            if limit != None:
                if gt_order[0] == 'asc':
                    order_par = 0
                else:
                    order_par = 1
            else:
                if gt_order[0] == 'asc':
                    order_par = 2
                else:
                    order_par = 3

        sql_temp['order'] = [order_aggs, order_cols, order_par] #GOLD for ORDER [[[agg], [col], [dat]], []]

        # process intersect/except/union
        sql_temp['special'] = 0
        if sql['sql']['intersect'] is not None:
            sql_temp['special'] = 1
        elif sql['sql']['except'] is not None:
            sql_temp['special'] = 2
        elif sql['sql']['union'] is not None:
            sql_temp['special'] = 3

        output_sql.append(sql_temp)

    return output_sql, output_tab
