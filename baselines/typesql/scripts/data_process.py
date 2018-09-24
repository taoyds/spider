import re
import io
import json
import numpy as np
import os

def load_data_new(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}
    for i, SQL_PATH in enumerate(sql_paths):
        if use_small and i >= 2:
            break
        print "Loading data from %s"%SQL_PATH
        with open(SQL_PATH) as inf:
            data = json.load(inf)
            sql_data += data
                
    for i, TABLE_PATH in enumerate(table_paths):
        if use_small and i >= 2:
            break
        print "Loading data from %s"%TABLE_PATH
        with open(TABLE_PATH) as inf:
            table_data= json.load(inf)
    # print sql_data[0]
    sql_data, table_data = process(sql_data, table_data)
    return sql_data, table_data

def process(sql_data, table_data):
	output_tab = {}
 	for i in range(len(table_data)):
 		table = table_data[i]
 		temp = {}
 		temp['col_map'] = table['column_names']

 		db_name = table['db_id']
 		output_tab[db_name] = temp


	output_sql = []
	for i in range(len(sql_data)):
		sql = sql_data[i]
		temp = {}

		# add query metadata
		temp['question'] = sql['question']
		temp['question_tok'] = sql['question_toks']
		temp['query'] = sql['query']
		temp['query_tok'] = sql['query_toks']
		temp['table_id'] = sql['db_id']
		sql_temp = {}

		# process agg/sel
		sql_temp['agg'] = []
		sql_temp['sel'] = []
 		gt_sel = sql['sql']['select'][1]
 		for tup in gt_sel:
 			sql_temp['agg'].append(tup[0])
 			sql_temp['sel'].append(tup[1][1][1])
		
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
				sql_temp['cond'].append(curr_cond)

		sql_temp['conj'] = [gt_cond[x] for x in range(len(gt_cond)) if x % 2 == 1]

		# process group by / having
		sql_temp['group'] = [x[1] for x in sql['sql']['groupBy']]
		having_cond = []
		if len(sql['sql']['having']) > 0:
			gt_having = sql['sql']['having'][0] # currently only do first having condition
			having_cond.append(gt_having[2][1][0]) # aggregator
			having_cond.append(gt_having[2][1][1]) # column
			having_cond.append(gt_having[1]) # operator
			if gt_having[4] is not None:
				having_cond.append([gt_having[3], gt_having[4]])
			else:
				having_cond.append(gt_having[3])
		sql_temp['group'].append(having_cond)

		# process order by / limit
		order_aggs = []
		order_cols = []
		order_par = -1
		gt_order = sql['sql']['orderBy']
		if len(gt_order) > 0:
			order_aggs = [x[1][0] for x in gt_order[1]]
			order_cols = [x[1][1] for x in gt_order[1]]
			order_par = 1 if gt_order[0] == 'asc' else 0
		sql_temp['order'] = [order_aggs, order_cols, order_par]

		# process intersect/except/union
		sql_temp['special'] = 0
		if sql['sql']['intersect'] is not None:
			sql_temp['special'] = 1
		elif sql['sql']['except'] is not None:
			sql_temp['special'] = 2
		elif sql['sql']['union'] is not None:
			sql_temp['special'] = 3

 		temp['sql'] = sql_temp
 		output_sql.append(temp)
	return output_sql, output_tab

if __name__ == '__main__':
    sql_data, table_data = load_data_new(['../../nl2sqlgit/data/train.json'], ['../../nl2sqlgit/data/tables.json'], False)
