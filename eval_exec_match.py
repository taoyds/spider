def col_unit_back(col_unit,tables_with_alias):
    if col_unit == None:
        return None

    bool_agg = False
    col = ""

    if col_unit[2]  != None and col_unit[2]:
        col = " distinct "
    if col_unit[0]  > 0:
        col = AGG_OPS[col_unit[0]] + '(' + col
        bool_agg = True

    name = col_unit[1]
    if name.endswith("__"):
        name = name[:-2]
    if name.startswith("__"):
        name = name[2:]
    if name == 'all':
        name = '*'

    nameArray = name.split('.')
    if len(nameArray) == 2:
        table_name = nameArray[0]
        for key,value in tables_with_alias.items():
            if key != table_name and value == table_name:
                name = key + "." + nameArray[1]
                break

    col = col + name

    if bool_agg:
        col = col + ')'

    return col



def val_unit_back(val_unit,tables_with_alias):
    val = ""
    if val_unit[0] > 0: # agg
        val = AGG_OPS[val_unit[0]] + '('

    col_1 = col_unit_back(val_unit[1][1], tables_with_alias)
    col_2 = col_unit_back(val_unit[1][2], tables_with_alias) 

    if val_unit[1][0] > 0 and col_2 != None: 
        val = val + col_1 + UNIT_OPS[val_unit[1][0]]+col_2
    else:
        val = val + col_1

    if val_unit[0] > 0:
        val = val + ')'
    return val

def create_orderBy(p_str, pred, tables_with_alias):
    if p_str.endswith(';'):
        p_str = p_str[:-1]
    p_str += " order By "

    if pred['select'][0] != None and pred['select'][0]:
        p_str = p_str + " distinct "

    for item in pred['select'][1]:
        p_str = p_str + val_unit_back(item, tables_with_alias) + " , "

    p_str = p_str[:-3] # delete " , "
    return p_str


def in_sub_sql(idx,toks):
    left = False
    right_num = 0
    for i in  range(idx-1,-1,-1):
        if toks[i] == '(':
            if right_num == 0:
                left = True
                break
            else:
                right_num -= 1
        if toks[i] == ')':
            right_num += 1
    if left:
        left_num = 0
        for i in  range(idx+1,len(toks),1):
            if toks[i] == ')':
                if left_num == 0:
                    return True
                else:
                    left_num -= 1
            if toks[i] == '(':
                left_num += 1
    return False


def find_main_select(idx,toks):
    m_s = []
    for i in range(idx-1,-1,-1):
        if toks[i].lower() == "select" and not in_sub_sql(i,toks):
            for j in range(i+1,idx,1):
                if toks[j].lower() == "from":
                    return m_s
                m_s.append(toks[j])
                
        if toks[i].lower() == "order" and not in_sub_sql(i,toks):
            return None
    return None


def create_orderby(sql_str, sql_dict):
    if sql_str.endswith(';'):
        sql_str = sql_str[:-1]

    toks = tokenize(sql_str)
    idx_order = []
    column_order = []                

    tok_s = find_main_select(len(toks)-1,toks)
    if tok_s:
        idx_order.append(len(toks)-1)
        column_order.append(tok_s)

    sql_plus_order = ""
    for idx, tok in enumerate(toks):
        sql_plus_order += tok + " " 

        if idx in idx_order:
            idx_ = idx_order.index(idx)
            sql_plus_order += " order By "
            for order in column_order[idx_]:
                if order.strip() != 'distinct' and order.strip() != '*':
                    sql_plus_order += order + " "
    if sql_plus_order.endswith(" order By "):
        sql_plus_order = sql_plus_order[:-10]
    return sql_plus_order

def eval_exec_match(db, schema, p_str, g_str, pred, gold, print_=False):
    """
    return 1 if the values between prediction and gold are matching
    in the corresponding index. Currently not support multiple col_unit(pairs).
    """
    
    if pred['orderBy'] == [] and gold['orderBy'] == []:
        try:
            p_str = create_orderby(p_str,pred)
            g_str = create_orderby(g_str,gold)
        except:
            return False
        pass
    
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(p_str)
        p_res = cursor.fetchall()
    except:
        if print_:
            print('You can not get any results from database. Because There is error in your SQL.')
        return False
    try:
        cursor.execute(g_str)
        q_res = cursor.fetchall()
    except:
        if print_:
            print('You can not get any results from database. Because Ground True SQL Error.')
            print(g_str)
        return False

    def res_map_origin(res, val_units):
        rmap = {}
        for idx, val_unit in enumerate(val_units):
            key = tuple(val_unit[1]) if not val_unit[2] else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
            rmap[key] = [r[idx] for r in res]
        return rmap
    
    def res_map_full(res, val_units):
        rmap = {}
        for idx, key in enumerate(val_units):
            #key = tuple(val_unit[1]) if not val_unit[2] else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
            rmap[str(key)] = [r[idx] for r in res]
        return rmap

    q_name_list = [unit[1][1][1] for unit in pred['select'][1]]
    if len(q_name_list)==len(set(q_name_list)):
        res_map = res_map_origin
        p_val_units = [unit[1] for unit in pred['select'][1]]
        q_val_units = [unit[1] for unit in gold['select'][1]]
    else:
        res_map = res_map_full
        p_val_units = [unit for unit in pred['select'][1]]
        q_val_units = [unit for unit in gold['select'][1]]

    if print_:
        print('predict result: ')
        print(p_res)
        print('correct result: ')
        print(q_res)
        print(res_map(p_res, p_val_units) == res_map(q_res, q_val_units))

    return res_map(p_res, p_val_units) == res_map(q_res, q_val_units)
