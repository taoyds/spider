'''
modified by dongxu wang from
Created on Jul 4, 2017

@author: Bhanu

'''
import numpy as np
from seq2seq.data import vocab
from gensim.models.keyedvectors import KeyedVectors
import json
import os.path
from nltk import word_tokenize
np.random.seed(12345)

VALUE_NUM_SYMBOL = "{value}"


def get_schema_vocab_mapping():
    column_map = {}
    table_map = {}
    with open( 'data/datasets/tables.json') as f:
        ex_list = json.load(f)
        for table_dict in ex_list:
            db_id = table_dict["db_id"]
            new_tokens = []
            column_names = table_dict["column_names"]
            table_names = table_dict["table_names"]
            column_names_original = table_dict["column_names_original"]
            table_names_original = table_dict["table_names_original"]
            for i in range(len(column_names)):
                item_original = column_names_original[i]
                item = column_names[i]
                column_map[item_original[1]] = table_names[int(item[0])].replace(" ", "_") + "^^^" +item[1].replace(" ", "_")
                
            for i in range(len(table_names)):
                item_original = table_names_original[i]
                item = table_names[i]
                table_map[item_original] = item.replace(" ", "_")
    return column_map, table_map



def get_average(words, model, separate_word_dict):
    
    vector = np.zeros(len(model['the']))
    
    for word in words:
        if word in model:
            vector += model[word]
            print "added %s to vector" % word
        else:
            if len(words) == 1 and word.lower().endswith("id"):
                vector += model["id"]
                if len(word) > 2 and word[0:-2] in model:
                    vector += model[word[0:-2]]
                print "%s not recognized; using id instead." % word
            else:
                if word in separate_word_dict:
                    vector += separate_word_dict[word]   
                else:
                    tmp_vec = np.random.uniform(-0.25, 0.25, size=len(model['the']))
                    vector += tmp_vec
                    separate_word_dict[word] = tmp_vec
                    print "%s not recognized; using random instead." % word 
    v_norm = np.linalg.norm(vector)
    if v_norm == 0:
        return vector
    return vector / v_norm

 
def get_word_vector(input_string, model, column_map, table_map, separate_word_dict):
    '''
    Given an input string and a gensim Word2Vec model, return a vector
    representation of the string. 
    '''
    if len(input_string) == 0:
        return np.zeros(len(model['the']))
    
    # Split on underscores and whitespace
    # change to input string
    
    if input_string in table_map:
        words = [w.lower() for w in table_map[input_string].split("_") if len(w) > 0]
        vector = get_average(words, model, separate_word_dict)
    elif input_string in column_map:
        table_name, column_name = column_map[input_string].split("^^^")
        table_words = [w.lower() for w in table_name.split("_") if len(w) > 0]
        column_words = [w.lower() for w in column_name.split("_") if len(w) > 0]
        vector = get_average(table_words, model, separate_word_dict) + get_average(column_words, model, separate_word_dict)
    else:
        words = [w.lower() for w in input_string.split("_") if len(w) > 0]
        vector = get_average(words, model, separate_word_dict)
    v_norm = np.linalg.norm(vector)    
    if v_norm == 0:
        return vector
    return vector / v_norm



def read_embed_from_file(filename, vocab_):
    word_dict = {}
    vecs = []
    with open(filename) as fin:
        for line in fin:
            parts = line.split('|||')
            word = parts[0]
            vec = parts[1].split()
            vecs.append(vec)
    embedding_mat = np.asarray(vecs, dtype=np.float32)
    return embedding_mat

def store_to_file(filename, vecs, vocab_):
    with open(filename, 'w') as fout:
        for i in range(len(vecs)):
            item = vecs[i]
            token = vocab_[i]
            item_list = item.tolist()
            vec = " ".join(str(v) for v in item_list)
            try:
                fout.write((token+"|||"+vec+ "\n"))
            except:
                fout.write((token+"|||"+vec+ "\n").encode("utf-8"))
    

def read_embeddings(embeddings_path, vocab_path, embed_dim, mode="source"):
    column_map, table_map = get_schema_vocab_mapping()
    filename = vocab_path.split(".")[0]+"saved_embedding_"+str(embed_dim) +"_"+ mode
    vocab_, _, _ = vocab.read_vocab(vocab_path)
    if os.path.isfile(filename):
        return read_embed_from_file(filename, vocab_)
    else:
        separate_word_dict = {}
        gensim_model = KeyedVectors.load_word2vec_format(embeddings_path, binary=False)
        # todo: how to deal with value
        separate_word_dict["{value}"] = np.random.uniform(-0.25, 0.25, size=len(gensim_model['the']))
        vecs = [get_word_vector(w, gensim_model, column_map, table_map, separate_word_dict) for w in vocab_]
        store_to_file(filename, vecs, vocab_)
        embedding_mat = np.asarray(vecs, dtype=np.float32)
    
    return embedding_mat


if __name__ == "__main__":
    column_map, table_map = get_schema_vocab_mapping()
    separate_word_dict = {}
    model = KeyedVectors.load_word2vec_format("data/glove/glove.6B.100d.txt", binary=False)
    
    for k in column_map.keys():
       print (k, column_map[k])
       get_word_vector(k, model, column_map, table_map, separate_word_dict)
        
    for k in table_map.keys():
        print (k, table_map[k])
#         get_word_vector(k, model, column_map, table_map, separate_word_dict)

