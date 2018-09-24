from nltk import word_tokenize
import json
import re
import logging

infiles_data_final = {'train': '../datasets/data_final/train.json',   
                'dev':'../datasets/data_final/dev.json',
                'schema': '../datasets/tables.json',
                'test': '../datasets/data_final/test.json'
}

infiles_data_radn_split = {'train': '../datasets/data_radn_split/train_radn.json',   
                           'dev': '../datasets/data_radn_split/dev_radn.json',
                           'schema': '../datasets/tables.json',
                           'test': '../datasets/data_radn_split/test_radn.json'
}

prefix_data_final = '../datasets/data_final_processed'
prefix_data_radn_split = '../datasets/data_radn_split_processed'
infiles = {}
infiles["data_final"] = [infiles_data_final, prefix_data_final]  
infiles["data_radn_split"] = [infiles_data_radn_split, prefix_data_radn_split]


VALUE_NUM_SYMBOL = "{value}"

def strip_table(table):
    column_types = table['column_types']
    table_names_original = [cn.lower() for cn in table['table_names_original']]
    table_names = [cn.lower() for cn in table['table_names']]
    column_names = [cn.lower() for i, cn in table['column_names']]
    column_names_original = [cn.lower() for i, cn in table['column_names_original']]
    return [table_names_original, table_names, column_names_original, column_names, column_types]



def strip_nl(nl):
    '''
    return keywords of nl query
    '''
    nl_keywords = []
    nl = nl.strip()
    nl = nl.replace(";"," ; ").replace(",", " , ").replace("?", " ? ").replace("\t"," ")
    nl = nl.replace("(", " ( ").replace(")", " ) ")
    
    str_1 = re.findall("\"[^\"]*\"", nl)
    str_2 = re.findall("\'[^\']*\'", nl)
    float_nums = re.findall("[-+]?\d*\.\d+", nl)
    
    values = str_1 + str_2 + float_nums
    for val in values:
        nl = nl.replace(val.strip(), VALUE_NUM_SYMBOL)
    
    
    raw_keywords = nl.strip().split()
    for tok in raw_keywords:
        if "." in tok:
            to = tok.replace(".", " . ").split()
            to = [t.lower() for t in to if len(t)>0]
            nl_keywords.extend(to)
        elif "'" in tok and tok[0]!="'" and tok[-1]!="'":
            to = word_tokenize(tok)
            to = [t.lower() for t in to if len(t)>0]
            nl_keywords.extend(to)      
        elif len(tok) > 0:
            nl_keywords.append(tok.lower())
    return nl_keywords


def strip_query(query):
    '''
    return keywords of sql query
    '''
    query_keywords = []
    query = query.strip().replace(";","").replace("\t","")
    query = query.replace("(", " ( ").replace(")", " ) ")
    query = query.replace(">=", " >= ").replace("<=", " <= ").replace("!=", " != ").replace("=", " = ")

    
    # then replace all stuff enclosed by "" with a numerical value to get it marked as {VALUE}
    str_1 = re.findall("\"[^\"]*\"", query)
    str_2 = re.findall("\'[^\']*\'", query)
    
    values = str_1 + str_2
    for val in values:
        query = query.replace(val.strip(), VALUE_NUM_SYMBOL)

    query_tokenized = query.split()
    float_nums = re.findall("[-+]?\d*\.\d+", query)
    query_tokenized = [VALUE_NUM_SYMBOL if qt in float_nums else qt for qt in query_tokenized]
    query = " ".join(query_tokenized)
    int_nums = [i.strip() for i in re.findall("[^tT]\d+", query)]

    
    query_tokenized = [VALUE_NUM_SYMBOL if qt in int_nums else qt for qt in query_tokenized]
    # print int_nums, query, query_tokenized
    
    for tok in query_tokenized:
        if "." in tok:
            table = re.findall("[Tt]\d+\.", tok)
            if len(table)>0:
                to = tok.replace(".", " . ").split()
                to = [t.lower() for t in to if len(t)>0]
                query_keywords.extend(to)
            else:
                query_keywords.append(tok.lower())

        elif len(tok) > 0:
            query_keywords.append(tok.lower())
    query_keywords = [w for w in query_keywords if len(w)>0]
    query_sentence = " ".join(query_keywords)
    query_sentence = query_sentence.replace("> =", ">=").replace("! =", "!=").replace("< =", "<=")
#     if '>' in query_sentence or '=' in query_sentence:
#        print query_sentence
    return query_sentence.split()





def count_databases(infile_group):
    content = set()
    with open(infile_group['dev']) as f:
        ex_list = json.load(f)
        for table_dict in ex_list:
            content.add(table_dict["db_id"])
    dev_count = len(content)
    print "the number of dev tables are", dev_count
    
    with open(infile_group['train']) as f:
        ex_list = json.load(f)
        for table_dict in ex_list:
            content.add(table_dict["db_id"])
    train_count = len(content) - dev_count
    print "the number of train tables are",train_count
    
    with open(infile_group['test']) as f: 
        ex_list = json.load(f)
        for table_dict in ex_list:
            db_id = table_dict["db_id"]
            if db_id not in table_dict:
                content.add(db_id)
    print "the number of total tables are", len(content)
    return content





def output_vocab_to_txt(outfile, cnt, min_frequency=0, max_vocab_size=None):
    file_obj = open(outfile, 'w')
    # output vocab with min_frequency, but the same weight
    logging.info("Found %d unique tokens in the vocabulary.", len(cnt))
    
    # Filter tokens below the frequency threshold
    fout_decode_file = open(outfile, 'w')
    if min_frequency > 0:
        filtered_tokens = [(w, c) for w, c in cnt.most_common()
                            if c > min_frequency]
        cnt = collections.Counter(dict(filtered_tokens))

        logging.info("Found %d unique tokens with frequency > %d.",
                     len(cnt), min_frequency)

    # Sort tokens by 1. frequency 2. lexically to break ties
    word_with_counts = cnt.most_common()
    word_with_counts = sorted(
        word_with_counts, key=lambda x: (x[1], x[0]), reverse=True)
    # Take only max-vocab
    if max_vocab_size is not None:
        word_with_counts = word_with_counts[:max_vocab_size]

    all_words = {}

    for word, count in word_with_counts:
        try:
            word = str(word)
            if word.strip() in all_words:
                pass
            else:
                all_words[word] = 1
                file_obj.write("{}\t{}\n".format(word, 1))                
        except:
                file_obj.write("{}\t{}\n".format(word.encode('utf-8'), 1))              
    file_obj.close()
    