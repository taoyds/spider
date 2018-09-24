# -*- coding: utf-8 -*- 
#! /usr/bin/env python
# adapted by dongxu wang from
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#pylint: disable=invalid-name
"""
Generate vocabulary for a tokenized text file.
"""


import sys
import argparse
import collections
import json
import os
from nltk.tokenize import word_tokenize
import re
from utils import output_vocab_to_txt, infiles, count_databases, strip_query, strip_nl

parser = argparse.ArgumentParser(
    description="Generate vocabulary for a tokenized text file.")
parser.add_argument(
    "--min_frequency",
    dest="min_frequency",
    type=int,
    default=0,
    help="Minimum frequency of a word to be included in the vocabulary.")
parser.add_argument(
    "--max_vocab_size",
    dest="max_vocab_size",
    type=int,
    help="Maximum number of tokens in the vocabulary")

args = parser.parse_args()
    

def get_encode_Query(infile_group, outfile, infile, output=False):
    max_nl = 0
    cnt = collections.Counter()
    infile = infile_group[infile]
    if output:
        outfile = open(outfile, 'w')
    
    cnt = collections.Counter()    
    with open(infile) as f:
        ex_list = json.load(f)
        for nl_sql_dict in ex_list:
            tokens = strip_nl(nl_sql_dict["question"])                    
            cnt.update(tokens)
            max_nl = max(max_nl, len(tokens))
            token_sentence = " ".join(tokens)
            if output:
                try:
                    outfile.write("{}\n".format(token_sentence))
                except:
                    outfile.write("{}\n".format(token_sentence.encode('utf-8')))
                
        if output:
            outfile.close()
    print "max length nl of", infile, "is", max_nl
    return cnt


def get_mask(infile_group, outfile, infile, vocabfile, output=True):
    
    
    
    _, used_databases, db_dict_rev = get_schema_vocab(infile_group, "schema")
    key_words = sql_key_words()
    infile_name = infile_group[infile]
    if output:
        outfile = open(outfile, 'w')
    vocab_ls = []
    with open(vocabfile, "r") as vocab:
        for line in vocab:
            items = line.split()
            vocab_ls.append(items[0])
    print "decode vocab length is", len(vocab_ls)
    with open(infile_name) as f:
        ex_list = json.load(f)
        for nl_sql_dict in ex_list:
            if infile != 'train':
                db_id = nl_sql_dict["db_id"]

                binary = []
                for item in vocab_ls:
                    # print db_id
                    if item in key_words:
                        binary.append("1")
                    elif (item in db_dict_rev ) and (db_id in db_dict_rev[item]):
                        binary.append("1")
                    else:
                        binary.append("0")
                binary.extend(["1"] * 5)
                if output:
                    outfile.write("{}\n".format(" ".join(binary)))
            elif output:
                outfile.write("{}\n".format("1"))
                

        if output:
            outfile.close()
        
    
        
def get_decode_SQL(infile_group, outfile, infile, output=False, outputdb=False):
    max_sql = 0
    
    cnt = collections.Counter()
    infile = infile_group[infile]
    if output:
        outfile = open(outfile, 'w')
    
    with open(infile) as f:
        ex_list = json.load(f)
        for nl_sql_dict in ex_list:
            tokens = strip_query(nl_sql_dict["query"])       
            max_sql = max(max_sql, len(tokens))
            cnt.update(tokens)
            token_sentence = " ".join(tokens)
            if output and not outputdb:
                try:
                    outfile.write("{}\n".format(token_sentence))
                except:
                    outfile.write("{}\n".format(token_sentence.encode('utf-8')))
            elif output and outputdb:
                try:
                    outfile.write("{}\t{}\n".format(nl_sql_dict["query"].lower().replace("\t", " "), nl_sql_dict["db_id"]))
                except:
                    outfile.write("{}\t{}\n".format(nl_sql_dict["query"].encode('utf-8').lower().replace("\t", " "), nl_sql_dict["db_id"]))
        if output:
            outfile.close()
    print "max sql length of", infile, "is", max_sql
    return cnt

def get_schema_vocab(infile_group, infile):
    used_databases = set()
    cnt = collections.Counter()
    db_dict_rev = {}
    with open(infile_group[infile]) as f:
        ex_list = json.load(f)
        for table_dict in ex_list:
            db_id = table_dict["db_id"]
            if db_id not in used_databases:
                used_databases.add(db_id)
            new_tokens = []
            column_names = table_dict["column_names_original"]
            table_names = table_dict["table_names_original"]
            for item in column_names:
                new_tokens.append(item[1].lower())
            for table_name in table_names:
                new_tokens.append(table_name.lower())
            cnt.update(new_tokens)   
            
            # build look up
            for tok in new_tokens:
                if tok not in db_dict_rev:
                    db_dict_rev[tok] = [db_id]
                else:
                    if db_id not in db_dict_rev[tok]:
                        db_dict_rev[tok].append(db_id)
                    
                
    return cnt, used_databases, db_dict_rev



def sql_key_words():
    cnt = collections.Counter()
    cnt.update(["t"+str(i+1) for i in range(10)])
    cnt.update([".", ",", "(", ")", "in", "not", "and", "between", "or", "where",
                "except", "union", "intersect",
                "group", "by", "order", "limit", "having","asc", "desc",
                "count", "sum", "avg", "max", "min",
                "{value}",
               "<", ">", "=", "!=", ">=", "<=",
                "like",
                "distinct","*",
                "join", "on", "as", "select", "from"
               ])
    return cnt


def get_decode_vocab_no_weight(infile_group, outfile):

    cnt, _, db_dict_rev = get_schema_vocab(infile_group, "schema")
    cnt.update(sql_key_words())
    output_vocab_to_txt(outfile, cnt)



def get_encode_vocab_no_weight(infile_group, outfile):
    cnt = collections.Counter()
    cnt.update(get_encode_Query(infile_group, None, 'train', False))
    cnt.update(get_encode_Query(infile_group, None, 'dev', False))   
    cnt.update(get_schema_vocab(infile_group, "schema")[0])
    output_vocab_to_txt(outfile, cnt)

    
def decode_encode_copy(infile_group, outfile):
    cnt = collections.Counter()

    cnt.update(get_encode_Query(infile_group, None, 'train', False))
    cnt.update(get_encode_Query(infile_group, None, 'dev', False))   
    cnt.update(get_schema_vocab(infile_group, "schema")[0])
    cnt.update(sql_key_words())

    output_vocab_to_txt(outfile, cnt)
    
    

    
if __name__ == "__main__":
    

    for key in infiles.keys():
        infile_group, prefix = infiles[key]
        count_databases(infile_group)
        if not os.path.exists(os.path.join(prefix,'dev')):
            os.makedirs(os.path.join(prefix,'dev'))
        if not os.path.exists(os.path.join(prefix,'train')):
            os.makedirs(os.path.join(prefix,'train'))
        if not os.path.exists(os.path.join(prefix,'test')):
            os.makedirs(os.path.join(prefix,'test'))

        get_decode_SQL(infile_group, os.path.join(prefix,'dev', "dev_decode.txt"), "dev", True)        
        get_decode_SQL(infile_group, os.path.join(prefix,'test', "test_decode.txt"), "test", True)
        get_decode_SQL(infile_group, os.path.join(prefix,'train', "train_decode.txt"), "train", True)
        
        get_decode_SQL(infile_group, os.path.join(prefix,'dev', "dev_decode_db.txt"), "dev", True, True)       
        get_decode_SQL(infile_group, os.path.join(prefix,'test', "test_decode_db.txt"), "test", True, True)
        get_decode_SQL(infile_group, os.path.join(prefix,'train', "train_decode_db.txt"), "train", True, True)

        get_encode_Query(infile_group, os.path.join(prefix,'test', "test_encode.txt"), "test", True)
        get_encode_Query(infile_group, os.path.join(prefix,'dev', "dev_encode.txt"), "dev", True)
        get_encode_Query(infile_group, os.path.join(prefix,'train', "train_encode.txt"), "train", True)

        get_encode_vocab_no_weight(infile_group, os.path.join(prefix, "encode_vocab.txt"))

        get_decode_vocab_no_weight(infile_group, os.path.join(prefix, "decode_vocab.txt"))
        get_decode_vocab_no_weight(infile_group, os.path.join(prefix, "decode_copy_encode_vocab.txt"))


        get_mask(infile_group, os.path.join(prefix, "test", "test_decoder_mask.txt"), "test",os.path.join(prefix, "decode_vocab.txt"), True)
        get_mask(infile_group, os.path.join(prefix, "dev", "dev_decoder_mask.txt"), "dev",os.path.join(prefix, "decode_vocab.txt"), True)
        get_mask(infile_group, os.path.join(prefix, "train", "train_decoder_mask.txt"), "train", os.path.join(prefix, "decode_vocab.txt"),True)


    
