import ast
import re
import sys, inspect
from StringIO import StringIO

import astor
from collections import OrderedDict
from tokenize import generate_tokens, tokenize
import token as tk

from nn.utils.io_utils import deserialize_from_file, serialize_to_file

from astnode import *


if __name__ == '__main__':
    #     node = ast.parse('''
    # # for i in range(1, 100):
    # #  sum = sum + i
    # #
    # # sorted(arr, reverse=True)
    # # sorted(my_dict, key=lambda x: my_dict[x], reverse=True)
    # # m = dict ( zip ( new_keys , keys ) )
    # # for f in sorted ( os . listdir ( self . path ) ) :
    # #     pass
    # for f in sorted ( os . listdir ( self . path ) ) : pass
    # ''')
    # print ast.dump(node, annotate_fields=False)
    # print get_tree_str_repr(node)
    # print parse('sorted(my_dict, key=lambda x: my_dict[x], reverse=True)')
    # print parse('global _standard_context_processors')

    # parse_django('/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code')

    # code = 'sum = True'
    # print parse_tree
    # ast_tree = tree_to_ast(parse_tree)
    # # # # #
    # import astor
    # print astor.to_source(ast_tree)

    from dataset import DataSet, Vocab, DataEntry, Action
    # train_data, dev_data, test_data = deserialize_from_file('django.cleaned.dataset.bin')
    # cand_list = deserialize_from_file('cand_hyps.18771.bin')
    # hyp_tree = cand_list[3].tree
    #
    # ast_tree = decode_tree_to_ast(hyp_tree)
    # print astor.to_source(ast_tree)

    pass