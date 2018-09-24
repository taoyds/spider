# -*- coding: UTF-8 -*-
from __future__ import division
import ast
import astor
import logging
from itertools import chain
import nltk
import re

from nn.utils.io_utils import serialize_to_file, deserialize_from_file
from nn.utils.generic_utils import init_logging

from dataset import gen_vocab, DataSet, DataEntry, Action, APPLY_RULE, GEN_TOKEN, COPY_TOKEN, GEN_COPY_TOKEN, Vocab
from lang.py.parse import parse, parse_tree_to_python_ast, canonicalize_code, get_grammar, parse_raw, \
    de_canonicalize_code, tokenize_code, tokenize_code_adv, de_canonicalize_code_for_seq2seq
from lang.py.unaryclosure import get_top_unary_closures, apply_unary_closures


def extract_grammar(code_file, prefix='py'):
    line_num = 0
    parse_trees = []
    for line in open(code_file):
        code = line.strip()
        parse_tree = parse(code)

        # leaves = parse_tree.get_leaves()
        # for leaf in leaves:
        #     if not is_terminal_type(leaf.type):
        #         print parse_tree

        # parse_tree = add_root(parse_tree)

        parse_trees.append(parse_tree)

        # sanity check
        ast_tree = parse_tree_to_python_ast(parse_tree)
        ref_ast_tree = ast.parse(canonicalize_code(code)).body[0]
        source1 = astor.to_source(ast_tree)
        source2 = astor.to_source(ref_ast_tree)

        assert source1 == source2

        # check rules
        # rule_list = parse_tree.get_rule_list(include_leaf=True)
        # for rule in rule_list:
        #     if rule.parent.type == int and rule.children[0].type == int:
        #         # rule.parent.type == str and rule.children[0].type == str:
        #         pass

        # ast_tree = tree_to_ast(parse_tree)
        # print astor.to_source(ast_tree)
            # print parse_tree
        # except Exception as e:
        #     error_num += 1
        #     #pass
        #     #print e

        line_num += 1

    print 'total line of code: %d' % line_num

    grammar = get_grammar(parse_trees)

    with open(prefix + '.grammar.txt', 'w') as f:
        for rule in grammar:
            str = rule.__repr__()
            f.write(str + '\n')

    with open(prefix + '.parse_trees.txt', 'w') as f:
        for tree in parse_trees:
            f.write(tree.__repr__() + '\n')

    return grammar, parse_trees


def rule_vs_node_stat():
    line_num = 0
    parse_trees = []
    code_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/card_datasets/hearthstone/all_hs.out' # '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code'
    node_nums = rule_nums = 0.
    for line in open(code_file):
        code = line.replace('§', '\n').strip()
        parse_tree = parse(code)
        node_nums += len(list(parse_tree.nodes))
        rules, _ = parse_tree.get_productions()
        rule_nums += len(rules)
        parse_trees.append(parse_tree)

        line_num += 1

    print 'avg. nums of nodes: %f' % (node_nums / line_num)
    print 'avg. nums of rules: %f' % (rule_nums / line_num)


def process_heart_stone_dataset():
    data_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/card_datasets/hearthstone/all_hs.out'
    parse_trees = []
    rule_num = 0.
    example_num = 0
    for line in open(data_file):
        code = line.replace('§', '\n').strip()
        parse_tree = parse(code)
        # sanity check
        pred_ast = parse_tree_to_python_ast(parse_tree)
        pred_code = astor.to_source(pred_ast)
        ref_ast = ast.parse(code)
        ref_code = astor.to_source(ref_ast)

        if pred_code != ref_code:
            raise RuntimeError('code mismatch!')

        rules, _ = parse_tree.get_productions(include_value_node=False)
        rule_num += len(rules)
        example_num += 1

        parse_trees.append(parse_tree)

    grammar = get_grammar(parse_trees)

    with open('hs.grammar.txt', 'w') as f:
        for rule in grammar:
            str = rule.__repr__()
            f.write(str + '\n')

    with open('hs.parse_trees.txt', 'w') as f:
        for tree in parse_trees:
            f.write(tree.__repr__() + '\n')


    print 'avg. nums of rules: %f' % (rule_num / example_num)


def canonicalize_hs_example(query, code):
    query = re.sub(r'<.*?>', '', query)
    query_tokens = nltk.word_tokenize(query)

    code = code.replace('§', '\n').strip()

    # sanity check
    parse_tree = parse_raw(code)
    gold_ast_tree = ast.parse(code).body[0]
    gold_source = astor.to_source(gold_ast_tree)
    ast_tree = parse_tree_to_python_ast(parse_tree)
    pred_source = astor.to_source(ast_tree)

    assert gold_source == pred_source, 'sanity check fails: gold=[%s], actual=[%s]' % (gold_source, pred_source)

    return query_tokens, code, parse_tree


def preprocess_hs_dataset(annot_file, code_file):
    f = open('hs_dataset.examples.txt', 'w')

    examples = []

    for idx, (annot, code) in enumerate(zip(open(annot_file), open(code_file))):
        annot = annot.strip()
        code = code.strip()

        clean_query_tokens, clean_code, parse_tree = canonicalize_hs_example(annot, code)
        example = {'id': idx, 'query_tokens': clean_query_tokens, 'code': clean_code, 'parse_tree': parse_tree,
                   'str_map': None, 'raw_code': code}
        examples.append(example)

        f.write('*' * 50 + '\n')
        f.write('example# %d\n' % idx)
        f.write(' '.join(clean_query_tokens) + '\n')
        f.write('\n')
        f.write(clean_code + '\n')
        f.write('*' * 50 + '\n')

        idx += 1

    f.close()

    print 'preprocess_dataset: cleaned example num: %d' % len(examples)

    return examples


def parse_hs_dataset():
    MAX_QUERY_LENGTH = 70 # FIXME: figure out the best config!
    WORD_FREQ_CUT_OFF = 3

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

    with open('hs.grammar.unary_closure.txt', 'w') as f:
        for rule in grammar:
            f.write(rule.__repr__() + '\n')

    annot_tokens = list(chain(*[e['query_tokens'] for e in data]))
    annot_vocab = gen_vocab(annot_tokens, vocab_size=5000, freq_cutoff=WORD_FREQ_CUT_OFF)

    def get_terminal_tokens(_terminal_str):
        """
        get terminal tokens
        break words like MinionCards into [Minion, Cards]
        """
        tmp_terminal_tokens = [t for t in _terminal_str.split(' ') if len(t) > 0]
        _terminal_tokens = []
        for token in tmp_terminal_tokens:
            sub_tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', token).split(' ')
            _terminal_tokens.extend(sub_tokens)

            _terminal_tokens.append(' ')

        return _terminal_tokens[:-1]

    # enumerate all terminal tokens to build up the terminal tokens vocabulary
    all_terminal_tokens = []
    for entry in data:
        parse_tree = entry['parse_tree']
        for node in parse_tree.get_leaves():
            if grammar.is_value_node(node):
                terminal_val = node.value
                terminal_str = str(terminal_val)

                terminal_tokens = get_terminal_tokens(terminal_str)

                for terminal_token in terminal_tokens:
                    assert len(terminal_token) > 0
                    all_terminal_tokens.append(terminal_token)

    terminal_vocab = gen_vocab(all_terminal_tokens, vocab_size=5000, freq_cutoff=WORD_FREQ_CUT_OFF)

    # now generate the dataset!

    train_data = DataSet(annot_vocab, terminal_vocab, grammar, 'hs.train_data')
    dev_data = DataSet(annot_vocab, terminal_vocab, grammar, 'hs.dev_data')
    test_data = DataSet(annot_vocab, terminal_vocab, grammar, 'hs.test_data')

    all_examples = []

    can_fully_reconstructed_examples_num = 0
    examples_with_empty_actions_num = 0

    for entry in data:
        idx = entry['id']
        query_tokens = entry['query_tokens']
        code = entry['code']
        parse_tree = entry['parse_tree']

        rule_list, rule_parents = parse_tree.get_productions(include_value_node=True)

        actions = []
        can_fully_reconstructed = True
        rule_pos_map = dict()

        for rule_count, rule in enumerate(rule_list):
            if not grammar.is_value_node(rule.parent):
                assert rule.value is None
                parent_rule = rule_parents[(rule_count, rule)][0]
                if parent_rule:
                    parent_t = rule_pos_map[parent_rule]
                else:
                    parent_t = 0

                rule_pos_map[rule] = len(actions)

                d = {'rule': rule, 'parent_t': parent_t, 'parent_rule': parent_rule}
                action = Action(APPLY_RULE, d)

                actions.append(action)
            else:
                assert rule.is_leaf

                parent_rule = rule_parents[(rule_count, rule)][0]
                parent_t = rule_pos_map[parent_rule]

                terminal_val = rule.value
                terminal_str = str(terminal_val)
                terminal_tokens = get_terminal_tokens(terminal_str)

                # assert len(terminal_tokens) > 0

                for terminal_token in terminal_tokens:
                    term_tok_id = terminal_vocab[terminal_token]
                    tok_src_idx = -1
                    try:
                        tok_src_idx = query_tokens.index(terminal_token)
                    except ValueError:
                        pass

                    d = {'literal': terminal_token, 'rule': rule, 'parent_rule': parent_rule, 'parent_t': parent_t}

                    # cannot copy, only generation
                    # could be unk!
                    if tok_src_idx < 0 or tok_src_idx >= MAX_QUERY_LENGTH:
                        action = Action(GEN_TOKEN, d)
                        if terminal_token not in terminal_vocab:
                            if terminal_token not in query_tokens:
                                # print terminal_token
                                can_fully_reconstructed = False
                    else:  # copy
                        if term_tok_id != terminal_vocab.unk:
                            d['source_idx'] = tok_src_idx
                            action = Action(GEN_COPY_TOKEN, d)
                        else:
                            d['source_idx'] = tok_src_idx
                            action = Action(COPY_TOKEN, d)

                    actions.append(action)

                d = {'literal': '<eos>', 'rule': rule, 'parent_rule': parent_rule, 'parent_t': parent_t}
                actions.append(Action(GEN_TOKEN, d))

        if len(actions) == 0:
            examples_with_empty_actions_num += 1
            continue

        example = DataEntry(idx, query_tokens, parse_tree, code, actions, {'str_map': None, 'raw_code': entry['raw_code']})

        if can_fully_reconstructed:
            can_fully_reconstructed_examples_num += 1

        # train, valid, test splits
        if 0 <= idx < 533:
            train_data.add(example)
        elif idx < 599:
            dev_data.add(example)
        else:
            test_data.add(example)

        all_examples.append(example)

    # print statistics
    max_query_len = max(len(e.query) for e in all_examples)
    max_actions_len = max(len(e.actions) for e in all_examples)

    # serialize_to_file([len(e.query) for e in all_examples], 'query.len')
    # serialize_to_file([len(e.actions) for e in all_examples], 'actions.len')

    logging.info('examples that can be fully reconstructed: %d/%d=%f',
                 can_fully_reconstructed_examples_num, len(all_examples),
                 can_fully_reconstructed_examples_num / len(all_examples))
    logging.info('empty_actions_count: %d', examples_with_empty_actions_num)

    logging.info('max_query_len: %d', max_query_len)
    logging.info('max_actions_len: %d', max_actions_len)

    train_data.init_data_matrices(max_query_length=70, max_example_action_num=350)
    dev_data.init_data_matrices(max_query_length=70, max_example_action_num=350)
    test_data.init_data_matrices(max_query_length=70, max_example_action_num=350)

    serialize_to_file((train_data, dev_data, test_data),
                      'data/hs.freq{WORD_FREQ_CUT_OFF}.max_action350.pre_suf.unary_closure.bin'.format(WORD_FREQ_CUT_OFF=WORD_FREQ_CUT_OFF))

    return train_data, dev_data, test_data


def dump_data_for_evaluation(data_type='django', data_file='', max_query_length=70):
    train_data, dev_data, test_data = deserialize_from_file(data_file)
    prefix = '/Users/yinpengcheng/Projects/dl4mt-tutorial/codegen_data/'
    for dataset, output in [(train_data, prefix + '%s.train' % data_type),
                            (dev_data, prefix + '%s.dev' % data_type),
                            (test_data, prefix + '%s.test' % data_type)]:
        f_source = open(output + '.desc', 'w')
        f_target = open(output + '.code', 'w')

        for e in dataset.examples:
            query_tokens = e.query[:max_query_length]
            code = e.code
            if data_type == 'django':
                target_code = de_canonicalize_code_for_seq2seq(code, e.meta_data['raw_code'])
            else:
                target_code = code

            # tokenize code
            target_code = target_code.strip()
            tokenized_target = tokenize_code_adv(target_code, breakCamelStr=False if data_type=='django' else True)
            tokenized_target = [tk.replace('\n', '#NEWLINE#') for tk in tokenized_target]
            tokenized_target = [tk for tk in tokenized_target if tk is not None]

            while tokenized_target[-1] == '#INDENT#':
                tokenized_target = tokenized_target[:-1]

            f_source.write(' '.join(query_tokens) + '\n')
            f_target.write(' '.join(tokenized_target) + '\n')

        f_source.close()
        f_target.close()


if __name__ == '__main__':
    init_logging('py.log')
    # rule_vs_node_stat()
    # process_heart_stone_dataset()
    parse_hs_dataset()
    # dump_data_for_evaluation(data_file='data/django.cleaned.dataset.freq5.par_info.refact.space_only.bin')
    # dump_data_for_evaluation(data_type='hs', data_file='data/hs.freq3.pre_suf.unary_closure.bin')
    # code_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code'
    # py_grammar, _ = extract_grammar(code_file)
    # serialize_to_file(py_grammar, 'py_grammar.bin')