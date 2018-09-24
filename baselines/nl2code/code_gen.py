import numpy as np
import cProfile
import ast
import traceback
import argparse
import os
import logging
from vprof import profiler

from model import Model
from dataset import DataEntry, DataSet, Vocab, Action
import config
from learner import Learner
from evaluation import *
from decoder import decode_python_dataset, decode_sql_dataset
from components import Hyp
from astnode import ASTNode

from nn.utils.generic_utils import init_logging
from nn.utils.io_utils import deserialize_from_file, serialize_to_file

parser = argparse.ArgumentParser()
parser.add_argument('-data')
parser.add_argument('-glove_path')
parser.add_argument('-ast_output_path',type=str)
parser.add_argument('-random_seed', default=181783, type=int)
parser.add_argument('-output_dir', default='.outputs')
parser.add_argument('-model', default=None)

# model's main configuration
parser.add_argument('-data_type', default='django', choices=['django', 'ifttt', 'hs','sql'])

# neural model's parameters
parser.add_argument('-source_vocab_size', default=0, type=int)
parser.add_argument('-target_vocab_size', default=0, type=int)
parser.add_argument('-rule_num', default=0, type=int)
parser.add_argument('-node_num', default=0, type=int)

parser.add_argument('-word_embed_dim', default=300, type=int)
parser.add_argument('-rule_embed_dim', default=256, type=int)
parser.add_argument('-node_embed_dim', default=256, type=int)
parser.add_argument('-encoder_hidden_dim', default=256, type=int)
parser.add_argument('-decoder_hidden_dim', default=256, type=int)
parser.add_argument('-attention_hidden_dim', default=50, type=int)
parser.add_argument('-ptrnet_hidden_dim', default=50, type=int)
parser.add_argument('-dropout', default=0.2, type=float)

# encoder
parser.add_argument('-encoder', default='bilstm', choices=['bilstm', 'lstm'])

# decoder
parser.add_argument('-parent_hidden_state_feed', dest='parent_hidden_state_feed', action='store_true')
parser.add_argument('-no_parent_hidden_state_feed', dest='parent_hidden_state_feed', action='store_false')
parser.set_defaults(parent_hidden_state_feed=True)

parser.add_argument('-parent_action_feed', dest='parent_action_feed', action='store_true')
parser.add_argument('-no_parent_action_feed', dest='parent_action_feed', action='store_false')
parser.set_defaults(parent_action_feed=True)

parser.add_argument('-frontier_node_type_feed', dest='frontier_node_type_feed', action='store_true')
parser.add_argument('-no_frontier_node_type_feed', dest='frontier_node_type_feed', action='store_false')
parser.set_defaults(frontier_node_type_feed=True)

parser.add_argument('-tree_attention', dest='tree_attention', action='store_true')
parser.add_argument('-no_tree_attention', dest='tree_attention', action='store_false')
parser.set_defaults(tree_attention=False)

parser.add_argument('-enable_copy', dest='enable_copy', action='store_true')
parser.add_argument('-no_copy', dest='enable_copy', action='store_false')
parser.set_defaults(enable_copy=True)

# training
parser.add_argument('-optimizer', default='adam')
parser.add_argument('-clip_grad', default=0., type=float)
parser.add_argument('-train_patience', default=10, type=int)
parser.add_argument('-max_epoch', default=50, type=int)
parser.add_argument('-batch_size', default=10, type=int)
parser.add_argument('-valid_per_batch', default=4000000000000, type=int)
parser.add_argument('-save_per_batch', default=400000000000000, type=int)
parser.add_argument('-valid_metric', default='bleu')

# decoding
parser.add_argument('-beam_size', default=15, type=int)
parser.add_argument('-max_query_length', default=70, type=int)
parser.add_argument('-decode_max_time_step', default=100, type=int)
parser.add_argument('-head_nt_constraint', dest='head_nt_constraint', action='store_true')
parser.add_argument('-no_head_nt_constraint', dest='head_nt_constraint', action='store_false')
parser.set_defaults(head_nt_constraint=True)
# parser.add_argument('-ast_output_path',type=str)
sub_parsers = parser.add_subparsers(dest='operation', help='operation to take')
train_parser = sub_parsers.add_parser('train')
decode_parser = sub_parsers.add_parser('decode')
interactive_parser = sub_parsers.add_parser('interactive')
evaluate_parser = sub_parsers.add_parser('evaluate')

# decoding operation
decode_parser.add_argument('-saveto', default='decode_results.bin')
decode_parser.add_argument('-type', default='test_data')

# evaluation operation
evaluate_parser.add_argument('-mode', default='self')
evaluate_parser.add_argument('-input', default='decode_results.bin')
evaluate_parser.add_argument('-type', default='test_data')
evaluate_parser.add_argument('-seq2tree_sample_file', default='model.sample')
evaluate_parser.add_argument('-seq2tree_id_file', default='test.id.txt')
evaluate_parser.add_argument('-seq2tree_rareword_map', default=None)
evaluate_parser.add_argument('-seq2seq_decode_file')
evaluate_parser.add_argument('-seq2seq_ref_file')
evaluate_parser.add_argument('-is_nbest', default=False, action='store_true')

# misc
parser.add_argument('-ifttt_test_split', default='data/ifff.test_data.gold.id')

# interactive operation
interactive_parser.add_argument('-mode', default='dataset')

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.random.seed(args.random_seed)
    init_logging(os.path.join(args.output_dir, 'parser.log'), logging.INFO)
    logging.info('command line: %s', ' '.join(sys.argv))

    logging.info('loading dataset [%s]', args.data)
    train_data, dev_data, test_data = deserialize_from_file(args.data)

    if not args.source_vocab_size:
        print("source vocab size".format(train_data.annot_vocab.size))
        args.source_vocab_size = train_data.annot_vocab.size
    if not args.target_vocab_size:
        print("target vocab size".format(train_data.terminal_vocab.size))
        args.target_vocab_size = train_data.terminal_vocab.size

    print("grammer:{}".format(len(train_data.grammar)))

    if not args.rule_num:
        args.rule_num = len(train_data.grammar.rules)
    if not args.node_num:
        args.node_num = len(train_data.grammar.node_type_to_id)

    logging.info('current config: %s', args)
    config_module = sys.modules['config']
    for name, value in vars(args).iteritems():
        setattr(config_module, name, value)

    # get dataset statistics
    avg_action_num = np.average([len(e.actions) for e in train_data.examples])
    logging.info('avg_action_num: %d', avg_action_num)

    logging.info('grammar rule num.: %d', len(train_data.grammar.rules))
    logging.info('grammar node type num.: %d', len(train_data.grammar.node_type_to_id))

    logging.info('source vocab size: %d', train_data.annot_vocab.size)
    logging.info('target vocab size: %d', train_data.terminal_vocab.size)

    if args.operation in ['train', 'decode', 'interactive']:
        model = Model(train_data.annot_vocab,args.glove_path)
        model.build()

        if args.model:
            model.load(args.model)

    if args.operation == 'train':
        # train_data = train_data.get_dataset_by_ids(range(2000), 'train_sample')
        # dev_data = dev_data.get_dataset_by_ids(range(10), 'dev_sample')
        learner = Learner(model, train_data, dev_data)
        learner.train()

    if args.operation == 'decode':
        # ==========================
        # investigate short examples
        # ==========================

        # short_examples = [e for e in test_data.examples if e.parse_tree.size <= 2]
        # for e in short_examples:
        #     print e.parse_tree
        # print 'short examples num: ', len(short_examples)

        # dataset = test_data # test_data.get_dataset_by_ids([1,2,3,4,5,6,7,8,9,10], name='sample')
        # cProfile.run('decode_dataset(model, dataset)', sort=2)

        # from evaluation import decode_and_evaluate_ifttt
        if args.data_type == 'ifttt':
            decode_results = decode_and_evaluate_ifttt_by_split(model, test_data)
        elif args.data_type == "sql":
            decode_results = decode_sql_dataset(model, test_data,args.ast_output_path)
        else:
            dataset = eval(args.type)
            decode_results = decode_python_dataset(model, dataset)

        serialize_to_file(decode_results, args.saveto)

    if args.operation == 'evaluate':
        dataset = eval(args.type)
        if config.mode == 'self':
            decode_results_file = args.input
            decode_results = deserialize_from_file(decode_results_file)

            evaluate_decode_results(dataset, decode_results)
        elif config.mode == 'seq2tree':
            from evaluation import evaluate_seq2tree_sample_file
            evaluate_seq2tree_sample_file(config.seq2tree_sample_file, config.seq2tree_id_file, dataset)
        elif config.mode == 'seq2seq':
            from evaluation import evaluate_seq2seq_decode_results
            evaluate_seq2seq_decode_results(dataset, config.seq2seq_decode_file, config.seq2seq_ref_file, is_nbest=config.is_nbest)
        elif config.mode == 'analyze':
            from evaluation import analyze_decode_results

            decode_results_file = args.input
            decode_results = deserialize_from_file(decode_results_file)
            analyze_decode_results(dataset, decode_results)

    if args.operation == 'interactive':
        from dataset import canonicalize_query, query_to_data
        from collections import namedtuple
        from lang.py.parse import decode_tree_to_python_ast
        assert model is not None

        while True:
            cmd = raw_input('example id or query: ')
            if args.mode == 'dataset':
                try:
                    example_id = int(cmd)
                    example = [e for e in test_data.examples if e.raw_id == example_id][0]
                except:
                    print 'something went wrong ...'
                    continue
            elif args.mode == 'new':
                # we play with new examples!
                query, str_map = canonicalize_query(cmd)
                vocab = train_data.annot_vocab
                query_tokens = query.split(' ')
                query_tokens_data = [query_to_data(query, vocab)]
                example = namedtuple('example', ['query', 'data'])(query=query_tokens, data=query_tokens_data)

            if hasattr(example, 'parse_tree'):
                print 'gold parse tree:'
                print example.parse_tree

            cand_list = model.decode(example, train_data.grammar, train_data.terminal_vocab,
                                     beam_size=args.beam_size, max_time_step=args.decode_max_time_step, log=True)

            has_grammar_error = any([c for c in cand_list if c.has_grammar_error])
            print 'has_grammar_error: ', has_grammar_error

            for cid, cand in enumerate(cand_list[:5]):
                print '*' * 60
                print 'cand #%d, score: %f' % (cid, cand.score)

                try:
                    ast_tree = decode_tree_to_python_ast(cand.tree)
                    code = astor.to_source(ast_tree)
                    print 'code: ', code
                    print 'decode log: ', cand.log
                except:
                    print "Exception in converting tree to code:"
                    print '-' * 60
                    print 'raw_id: %d, beam pos: %d' % (example.raw_id, cid)
                    traceback.print_exc(file=sys.stdout)
                    print '-' * 60
                finally:
                    print '* parse tree *'
                    print cand.tree.__repr__()
                    print 'n_timestep: %d' % cand.n_timestep
                    print 'ast size: %d' % cand.tree.size
                    print '*' * 60
