# -*- coding: UTF-8 -*-
from __future__ import division
import string
from collections import OrderedDict
from collections import defaultdict
from itertools import count

from nn.utils.io_utils import serialize_to_file, deserialize_from_file

from lang.ifttt.grammar import IFTTTGrammar
from parse import ifttt_ast_to_parse_tree
from lang.grammar import Grammar
import logging
from itertools import chain

from nn.utils.generic_utils import init_logging

from dataset import gen_vocab, DataSet, DataEntry, Action, APPLY_RULE, GEN_TOKEN, COPY_TOKEN, GEN_COPY_TOKEN

def load_examples(data_file):
    f = open(data_file)
    next(f)
    examples = []
    for line in f:
        d = line.strip().split('\t')
        description = d[4]
        code = d[9]
        parse_tree = ifttt_ast_to_parse_tree(code)

        examples.append({'description': description, 'parse_tree': parse_tree, 'code': code})

    return examples


def analyze_ifttt_dataset():
    data_file = '/Users/yinpengcheng/Research/SemanticParsing/ifttt/recipe_summaries.all.tsv'
    examples = load_examples(data_file)

    rule_num = 0.
    max_rule_num = -1
    example_with_max_rule_num = -1

    for idx, example in enumerate(examples):
        parse_tree = example['parse_tree']
        rules, _ = parse_tree.get_productions(include_value_node=True)

        rule_num += len(rules)
        if max_rule_num < len(rules):
            max_rule_num = len(rules)
            example_with_max_rule_num = idx

    logging.info('avg. num. of rules: %f', rule_num / len(examples))
    logging.info('max_rule_num: %d', max_rule_num)
    logging.info('example_with_max_rule_num: %d', example_with_max_rule_num)


def canonicalize_ifttt_example(annot, code):
    parse_tree = ifttt_ast_to_parse_tree(code, attach_func_to_channel=False)
    clean_code = str(parse_tree)
    clean_query_tokens = annot.split()
    clean_query_tokens = [t.lower() for t in clean_query_tokens]

    return clean_query_tokens, clean_code, parse_tree


def preprocess_ifttt_dataset(annot_file, code_file):
    f = open('ifttt_dataset.examples.txt', 'w')

    examples = []

    for idx, (annot, code) in enumerate(zip(open(annot_file), open(code_file))):
        annot = annot.strip()
        code = code.strip()

        clean_query_tokens, clean_code, parse_tree = canonicalize_ifttt_example(annot, code)
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


def get_grammar(parse_trees):
    rules = set()

    for parse_tree in parse_trees:
        parse_tree_rules, rule_parents = parse_tree.get_productions()
        for rule in parse_tree_rules:
            rules.add(rule)

    rules = list(sorted(rules, key=lambda x: x.__repr__()))
    grammar = IFTTTGrammar(rules)

    logging.info('num. rules: %d', len(rules))

    with open('grammar.txt', 'w') as f:
        for rule in grammar:
            str = rule.__repr__()
            f.write(str + '\n')

    with open('parse_trees.txt', 'w') as f:
        for tree in parse_trees:
            f.write(tree.__repr__() + '\n')

    return grammar


def parse_ifttt_dataset():
    WORD_FREQ_CUT_OFF = 2

    annot_file = '/Users/yinpengcheng/Research/SemanticParsing/ifttt/Data/lang.all.txt'
    code_file = '/Users/yinpengcheng/Research/SemanticParsing/ifttt/Data/code.all.txt'

    data = preprocess_ifttt_dataset(annot_file, code_file)

    # build the grammar
    grammar = get_grammar([e['parse_tree'] for e in data])

    annot_tokens = list(chain(*[e['query_tokens'] for e in data]))
    annot_vocab = gen_vocab(annot_tokens, vocab_size=30000, freq_cutoff=WORD_FREQ_CUT_OFF)

    logging.info('annot vocab. size: %d', annot_vocab.size)

    # we have no terminal tokens in ifttt
    all_terminal_tokens = []
    terminal_vocab = gen_vocab(all_terminal_tokens, vocab_size=4000, freq_cutoff=WORD_FREQ_CUT_OFF)

    # now generate the dataset!

    train_data = DataSet(annot_vocab, terminal_vocab, grammar, 'ifttt.train_data')
    dev_data = DataSet(annot_vocab, terminal_vocab, grammar, 'ifttt.dev_data')
    test_data = DataSet(annot_vocab, terminal_vocab, grammar, 'ifttt.test_data')

    all_examples = []

    can_fully_reconstructed_examples_num = 0
    examples_with_empty_actions_num = 0

    for entry in data:
        idx = entry['id']
        query_tokens = entry['query_tokens']
        code = entry['code']
        parse_tree = entry['parse_tree']

        # check if query tokens are valid
        query_token_ids = [annot_vocab[token] for token in query_tokens if token not in string.punctuation]
        valid_query_tokens_ids = [tid for tid in query_token_ids if tid != annot_vocab.unk]

        # remove examples with rare words from train and dev, avoid overfitting
        if len(valid_query_tokens_ids) == 0 and 0 <= idx < 77495 + 5171:
            continue

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
                raise RuntimeError('no terminals should be in ifttt dataset!')

        if len(actions) == 0:
            examples_with_empty_actions_num += 1
            continue

        example = DataEntry(idx, query_tokens, parse_tree, code, actions,
                            {'str_map': None, 'raw_code': entry['raw_code']})

        if can_fully_reconstructed:
            can_fully_reconstructed_examples_num += 1

        # train, valid, test splits
        if 0 <= idx < 77495:
            train_data.add(example)
        elif idx < 77495 + 5171:
            dev_data.add(example)
        else:
            test_data.add(example)

        all_examples.append(example)

    # print statistics
    max_query_len = max(len(e.query) for e in all_examples)
    max_actions_len = max(len(e.actions) for e in all_examples)

    # serialize_to_file([len(e.query) for e in all_examples], 'query.len')
    # serialize_to_file([len(e.actions) for e in all_examples], 'actions.len')

    logging.info('train_data examples: %d', train_data.count)
    logging.info('dev_data examples: %d', dev_data.count)
    logging.info('test_data examples: %d', test_data.count)

    logging.info('examples that can be fully reconstructed: %d/%d=%f',
                 can_fully_reconstructed_examples_num, len(all_examples),
                 can_fully_reconstructed_examples_num / len(all_examples))
    logging.info('empty_actions_count: %d', examples_with_empty_actions_num)

    logging.info('max_query_len: %d', max_query_len)
    logging.info('max_actions_len: %d', max_actions_len)

    train_data.init_data_matrices(max_query_length=40, max_example_action_num=6)
    dev_data.init_data_matrices()
    test_data.init_data_matrices()

    serialize_to_file((train_data, dev_data, test_data),
                      'data/ifttt.freq{WORD_FREQ_CUT_OFF}.bin'.format(WORD_FREQ_CUT_OFF=WORD_FREQ_CUT_OFF))

    return train_data, dev_data, test_data


def parse_data_for_seq2seq(data_file='data/ifttt.freq3.bin'):
    train_data, dev_data, test_data = deserialize_from_file(data_file)
    prefix = 'data/seq2seq/'

    for dataset, output in [(train_data, prefix + 'ifttt.train'),
                            (dev_data, prefix + 'ifttt.dev'),
                            (test_data, prefix + 'ifttt.test')]:
        f_source = open(output + '.desc', 'w')
        f_target = open(output + '.code', 'w')

        if 'test' in output:
            raw_ids = [int(i.strip()) for i in open('data/ifff.test_data.gold.id')]
            eids = [i for i, e in enumerate(test_data.examples) if e.raw_id in raw_ids]
            dataset = test_data.get_dataset_by_ids(eids, test_data.name + '.subset')

        for e in dataset.examples:
            query_tokens = e.query
            trigger = e.parse_tree['TRIGGER'].children[0].type + ' . ' + e.parse_tree['TRIGGER'].children[0].children[0].type
            action = e.parse_tree['ACTION'].children[0].type + ' . ' + e.parse_tree['ACTION'].children[0].children[0].type
            code = 'IF ' + trigger + ' THEN ' + action

            f_source.write(' '.join(query_tokens) + '\n')
            f_target.write(code + '\n')

        f_source.close()
        f_target.close()


def extract_turk_data():
    turk_annot_file = '/Users/yinpengcheng/Research/SemanticParsing/ifttt/public_release/data/turk_public.tsv'
    reference_file = '/Users/yinpengcheng/Research/SemanticParsing/ifttt/public_release/data/ifttt_public.tsv'

    f_turk = open(turk_annot_file)
    next(f_turk)

    annot_data = OrderedDict()
    for line in f_turk:
        d = line.strip().split('\t')
        url = d[0]
        if url not in annot_data:
            annot_data[url] = list()

        annot_data[url].append({'trigger_channel': d[2], 'trigger_func': d[3], 'action_channel': d[4], 'action_func': d[5]})

    f_ref = open(reference_file)
    next(f_ref)
    ref_data = OrderedDict()
    for line in f_ref:
        d = line.strip().split('\t')
        url = d[0]

        ref_data[url] = {'trigger_channel': d[2], 'trigger_func': d[3], 'action_channel': d[4], 'action_func': d[5]}

    lt_three_agree_with_gold = []
    non_english_examples = []
    unintelligible_examples = []
    for url, annots in annot_data.iteritems():
        vote_dict = defaultdict(int)
        ref = ref_data[url]
        match_with_gold_num = 0
        non_english_num = unintelligible_num = 0
        non_english_annots = []
        unintelligible_annots = []

        for annot in annots:
            if annot['trigger_channel'] == ref['trigger_channel'] and annot['trigger_func'] == ref['trigger_func'] and \
                annot['action_channel'] == ref['action_channel'] and annot['action_func'] == ref['action_func']:
                match_with_gold_num += 1
            vote_dict['#'.join(annot.values())] += 1

        for i, annot in enumerate(annots):
            if annot['trigger_channel'] == 'nonenglish' and annot['trigger_func'] == 'nonenglish' and \
                annot['action_channel'] == 'nonenglish' and annot['action_func'] == 'nonenglish':
                non_english_num += 1
                non_english_annots.append(i)

            if annot['trigger_channel'] == 'unintelligible' and annot['trigger_func'] == 'unintelligible' and \
                annot['action_channel'] == 'unintelligible' and annot['action_func'] == 'unintelligible':
                unintelligible_num += 1
                unintelligible_annots.append(i)

        max_vote_num = max(vote_dict.values())

        # omitting descriptions marked as non-English by a majority of the crowdsourced workers
        if non_english_num == max_vote_num:
            non_english_examples.append(url)

        non_english_and_unintelligible_num = len(set(non_english_annots).union(set(unintelligible_annots)))
        # if this example has no non_english and unintelligible annotations
        if non_english_and_unintelligible_num > 0: # < len(annots) - non_english_and_unintelligible_num:
            unintelligible_examples.append(url)

        if match_with_gold_num >= 3:
            lt_three_agree_with_gold.append(url)

    omit_non_english_examples = set(annot_data) - set(non_english_examples)
    omit_unintelligible_examples = set(annot_data) - set(unintelligible_examples)
    print len(omit_non_english_examples) # should be 3,741
    print len(omit_unintelligible_examples) # should be 2,262
    print len(lt_three_agree_with_gold) # should be 758

    url2id = defaultdict(count(0).next)
    for url in ref_data:
        url2id[url] = url2id[url] + 77495 + 5171

    f_gold = open('data/ifff.test_data.gold.id', 'w')
    for url in lt_three_agree_with_gold:
        i = url2id[url]
        f_gold.write(str(i) + '\n')
    f_gold.close()

    f_gold = open('data/ifff.test_data.omit_unintelligible.id', 'w')
    for url in omit_unintelligible_examples:
        i = url2id[url]
        f_gold.write(str(i) + '\n')
    f_gold.close()

    f_gold = open('data/ifff.test_data.omit_non_english.id', 'w')
    for url in omit_non_english_examples:
        i = url2id[url]
        f_gold.write(str(i) + '\n')
    f_gold.close()

    omit_non_english_examples = [url2id[url] for url in omit_non_english_examples]
    omit_unintelligible_examples = [url2id[url] for url in omit_unintelligible_examples]
    lt_three_agree_with_gold = [url2id[url] for url in lt_three_agree_with_gold]

    return omit_non_english_examples, omit_unintelligible_examples, lt_three_agree_with_gold

if __name__ == '__main__':
    init_logging('ifttt.log')
    # parse_ifttt_dataset()
    # analyze_ifttt_dataset()
    extract_turk_data()
    # parse_data_for_seq2seq()
