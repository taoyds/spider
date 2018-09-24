import logging
import re
from collections import defaultdict, OrderedDict
from itertools import chain

import sys

from astnode import ASTNode
from dataset import preprocess_dataset, gen_vocab
from lang.py.grammar import type_str_to_type
from lang.py.parse import parse, get_grammar, decode_tree_to_python_ast
from lang.py.unaryclosure import get_top_unary_closures, apply_unary_closures
from lang.util import typename, escape, unescape
from nn.utils.generic_utils import init_logging
from nn.utils.io_utils import serialize_to_file


def ast_tree_to_seq2tree_repr(tree):
    repr_str = ''

    # node_name = typename(tree.type)
    label_val = '' if tree.label is None else tree.label
    value = '' if tree.value is None else tree.value
    node_name = '%s{%s}{%s}' % (typename(tree.type), label_val, value)
    repr_str += node_name

    # wrap children with parentheses
    if tree.children:
        repr_str += ' ('

        for child in tree.children:
            child_repr = ast_tree_to_seq2tree_repr(child)
            repr_str += ' ' + child_repr

        repr_str += ' )'

    return repr_str

node_re = re.compile(r'(?P<type>.*?)\{(?P<label>.*?)\}\{(?P<value>.*)\}')
def seq2tree_repr_to_ast_tree_helper(tree_repr, offset):
    """convert a seq2tree representation to AST tree"""

    # extract node name
    node_name_end = offset
    while node_name_end < len(tree_repr) and tree_repr[node_name_end] != ' ':
        node_name_end += 1

    node_repr = tree_repr[offset:node_name_end]

    m = node_re.match(node_repr)
    n_type = m.group('type')
    n_type = type_str_to_type(n_type)
    n_label = m.group('label')
    n_value = m.group('value')

    if n_type in {int, float, str, bool}:
        n_value = n_type(n_value)

    n_label = None if n_label == '' else n_label
    n_value = None if n_value == '' else n_value

    node = ASTNode(n_type, label=n_label, value=n_value)
    offset = node_name_end

    if offset == len(tree_repr):
        return node, offset

    offset += 1
    if tree_repr[offset] == '(':
        offset += 2
        while True:
            child_node, offset = seq2tree_repr_to_ast_tree_helper(tree_repr, offset=offset)
            node.add_child(child_node)

            if offset >= len(tree_repr) or tree_repr[offset] == ')':
                offset += 2
                break

    return node, offset


def seq2tree_repr_to_ast_tree(tree_repr):
    tree, _ = seq2tree_repr_to_ast_tree_helper(tree_repr, 0)

    return tree


def break_value_nodes(tree, hs=False):
    """inplace break value nodes with a string separaed by spaces"""
    if tree.type == str and tree.value is not None:
        assert tree.is_leaf

        if hs:
            tokens = re.sub(r'([a-z])([A-Z])', r'\1 #MERGE# \2', tree.value).split(' ')
        else:
            tokens = tree.value.split(' ')
        tree.value = 'NT'
        for token in tokens:
            assert token is not None
            tree.add_child(ASTNode(tree.type, value=escape(token)))
    else:
        for child in tree.children:
            break_value_nodes(child, hs=hs)


def merge_broken_value_nodes(tree):
    """redo *break_value_nodes*"""
    if tree.type == str and not tree.is_leaf:
        assert tree.value == 'NT'

        valid_children = [c for c in tree.children if c.value is not None]
        value = ' '.join(unescape(c.value) for c in valid_children)
        value = value.replace(' #MERGE# ', '')
        tree.value = value

        tree.children = []
    else:
        for child in tree.children:
            merge_broken_value_nodes(child)


def parse_django_dataset_for_seq2tree():
    from lang.py.parse import parse_raw
    MAX_QUERY_LENGTH = 70
    MAX_DECODING_TIME_STEP = 300
    UNARY_CUTOFF_FREQ = 30

    annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno'
    code_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code'

    data = preprocess_dataset(annot_file, code_file)

    for e in data:
        e['parse_tree'] = parse_raw(e['code'])

    parse_trees = [e['parse_tree'] for e in data]

    # apply unary closures
    # unary_closures = get_top_unary_closures(parse_trees, k=0, freq=UNARY_CUTOFF_FREQ)
    # for i, parse_tree in enumerate(parse_trees):
    #     apply_unary_closures(parse_tree, unary_closures)

    # build the grammar
    grammar = get_grammar(parse_trees)

    # # build grammar ...
    # from lang.py.py_dataset import extract_grammar
    # grammar, all_parse_trees = extract_grammar(code_file)

    f_train = open('/Users/yinpengcheng/Research/lang2logic/seq2tree/django/data/train.txt', 'w')
    f_dev = open('/Users/yinpengcheng/Research/lang2logic/seq2tree/django/data/dev.txt', 'w')
    f_test = open('/Users/yinpengcheng/Research/lang2logic/seq2tree/django/data/test.txt', 'w')

    f_train_rawid = open('/Users/yinpengcheng/Research/lang2logic/seq2tree/django/data/train.id.txt', 'w')
    f_dev_rawid = open('/Users/yinpengcheng/Research/lang2logic/seq2tree/django/data/dev.id.txt', 'w')
    f_test_rawid = open('/Users/yinpengcheng/Research/lang2logic/seq2tree/django/data/test.id.txt', 'w')

    decode_time_steps = defaultdict(int)

    # first pass
    for entry in data:
        idx = entry['id']
        query_tokens = entry['query_tokens']
        code = entry['code']
        parse_tree = entry['parse_tree']

        original_parse_tree = parse_tree.copy()
        break_value_nodes(parse_tree)
        tree_repr = ast_tree_to_seq2tree_repr(parse_tree)

        num_decode_time_step = len(tree_repr.split(' '))
        decode_time_steps[num_decode_time_step] += 1

        new_tree = seq2tree_repr_to_ast_tree(tree_repr)
        merge_broken_value_nodes(new_tree)

        query_tokens = [t for t in query_tokens if t != ''][:MAX_QUERY_LENGTH]
        query = ' '.join(query_tokens)
        line = query + '\t' + tree_repr

        if num_decode_time_step > MAX_DECODING_TIME_STEP:
            continue

        # train, valid, test
        if 0 <= idx < 16000:
            f_train.write(line + '\n')
            f_train_rawid.write(str(idx) + '\n')
        elif 16000 <= idx < 17000:
            f_dev.write(line + '\n')
            f_dev_rawid.write(str(idx) + '\n')
        else:
            f_test.write(line + '\n')
            f_test_rawid.write(str(idx) + '\n')

        if original_parse_tree != new_tree:
            print '*' * 50
            print idx
            print code

    f_train.close()
    f_dev.close()
    f_test.close()

    f_train_rawid.close()
    f_dev_rawid.close()
    f_test_rawid.close()

    # print 'num. of decoding time steps distribution:'
    # for k in sorted(decode_time_steps):
    #     print '%d\t%d' % (k, decode_time_steps[k])


def parse_hs_dataset_for_seq2tree():
    from lang.py.py_dataset import preprocess_hs_dataset
    MAX_QUERY_LENGTH = 70 # FIXME: figure out the best config!
    WORD_FREQ_CUT_OFF = 3
    MAX_DECODING_TIME_STEP = 800

    annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/card_datasets/hearthstone/all_hs.mod.in'
    code_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/card_datasets/hearthstone/all_hs.out'

    data = preprocess_hs_dataset(annot_file, code_file)
    parse_trees = [e['parse_tree'] for e in data]

    # apply unary closures
    unary_closures = get_top_unary_closures(parse_trees, k=20)
    for parse_tree in parse_trees:
        apply_unary_closures(parse_tree, unary_closures)

    # build the grammar
    grammar = get_grammar(parse_trees)

    decode_time_steps = defaultdict(int)

    f_train = open('/Users/yinpengcheng/Research/lang2logic/seq2tree/hs/data_unkreplaced/train.txt', 'w')
    f_dev = open('/Users/yinpengcheng/Research/lang2logic/seq2tree/hs/data_unkreplaced/dev.txt', 'w')
    f_test = open('/Users/yinpengcheng/Research/lang2logic/seq2tree/hs/data_unkreplaced/test.txt', 'w')

    f_train_rawid = open('/Users/yinpengcheng/Research/lang2logic/seq2tree/hs/data_unkreplaced/train.id.txt', 'w')
    f_dev_rawid = open('/Users/yinpengcheng/Research/lang2logic/seq2tree/hs/data_unkreplaced/dev.id.txt', 'w')
    f_test_rawid = open('/Users/yinpengcheng/Research/lang2logic/seq2tree/hs/data_unkreplaced/test.id.txt', 'w')

    # first pass
    for entry in data:
        idx = entry['id']
        query_tokens = entry['query_tokens']
        parse_tree = entry['parse_tree']

        original_parse_tree = parse_tree.copy()
        break_value_nodes(parse_tree, hs=True)
        tree_repr = ast_tree_to_seq2tree_repr(parse_tree)

        num_decode_time_step = len(tree_repr.split(' '))
        decode_time_steps[num_decode_time_step] += 1

        new_tree = seq2tree_repr_to_ast_tree(tree_repr)
        merge_broken_value_nodes(new_tree)

        query_tokens = [t for t in query_tokens if t != ''][:MAX_QUERY_LENGTH]
        query = ' '.join(query_tokens)
        line = query + '\t' + tree_repr

        if num_decode_time_step > MAX_DECODING_TIME_STEP:
            continue

        # train, valid, test
        if 0 <= idx < 533:
            f_train.write(line + '\n')
            f_train_rawid.write(str(idx) + '\n')
        elif idx < 599:
            f_dev.write(line + '\n')
            f_dev_rawid.write(str(idx) + '\n')
        else:
            f_test.write(line + '\n')
            f_test_rawid.write(str(idx) + '\n')

        if original_parse_tree != new_tree:
            print '*' * 50
            print idx
            print code

    f_train.close()
    f_dev.close()
    f_test.close()

    f_train_rawid.close()
    f_dev_rawid.close()
    f_test_rawid.close()

    # print 'num. of decoding time steps distribution:'
    for k in sorted(decode_time_steps):
        print '%d\t%d' % (k, decode_time_steps[k])


if __name__ == '__main__':
    init_logging('py.log')
    # code = "return (  format_html_join ( '' , '_STR:0_' , sorted ( attrs . items ( ) ) ) +  format_html_join ( '' , ' {0}' , sorted ( boolean_attrs ) )  )"
    code = "call('{0}')"
    parse_tree = parse(code)

    # parse_tree = ASTNode('root', children=[
    #     ASTNode('lambda'),
    #     ASTNode('$0'),
    #     ASTNode('e', children=[
    #         ASTNode('and', children=[
    #             ASTNode('>', children=[ASTNode('$0')]),
    #             ASTNode('from', children=[ASTNode('$0'), ASTNode('ci0')]),
    #         ])
    #     ]),
    # ])

    original_parse_tree = parse_tree.copy()
    break_value_nodes(parse_tree)

    # tree_repr = """root{}{} ( For{}{} ( expr{target}{} ( Name{}{} ( str{id}{NT} ( ) ) ) expr{iter}{} ( Name{}{} ( str{id}{NT} ( Name{}{} ( str{id}{NT} ( str{}{self} ) ) ) ) ) stmt*{body}{} ( stmt{}{} ( Pass{}{} ) ) ) )"""
    # print tree_repr

    # new_tree = seq2tree_repr_to_ast_tree(tree_repr)
    # merge_broken_value_nodes(new_tree)

    # print str(original_parse_tree)
    # print str(new_tree)

    # assert original_parse_tree == new_tree

    # parse_django_dataset_for_seq2tree()
    parse_hs_dataset_for_seq2tree()