import argparse, sys
from nn.utils.generic_utils import init_logging
from nn.utils.io_utils import deserialize_from_file, serialize_to_file
from evaluation import *
from dataset import canonicalize_query, query_to_data
from collections import namedtuple
from lang.py.parse import decode_tree_to_python_ast
from model import Model
from dataset import DataEntry, DataSet, Vocab, Action
import config

parser = argparse.ArgumentParser()
parser.add_argument('-data_type', default='django', choices=['django', 'hs'])
parser.add_argument('-data')
parser.add_argument('-random_seed', default=181783, type=int)
parser.add_argument('-model', default=None)

# neural model's parameters
parser.add_argument('-source_vocab_size', default=0, type=int)
parser.add_argument('-target_vocab_size', default=0, type=int)
parser.add_argument('-rule_num', default=0, type=int)
parser.add_argument('-node_num', default=0, type=int)

parser.add_argument('-word_embed_dim', default=128, type=int)
parser.add_argument('-rule_embed_dim', default=128, type=int)
parser.add_argument('-node_embed_dim', default=64, type=int)
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
parser.add_argument('-valid_per_batch', default=4000, type=int)
parser.add_argument('-save_per_batch', default=4000, type=int)
parser.add_argument('-valid_metric', default='bleu')

# decoding
parser.add_argument('-beam_size', default=15, type=int)
parser.add_argument('-max_query_length', default=70, type=int)
parser.add_argument('-decode_max_time_step', default=100, type=int)
parser.add_argument('-head_nt_constraint', dest='head_nt_constraint', action='store_true')
parser.add_argument('-no_head_nt_constraint', dest='head_nt_constraint', action='store_false')
parser.set_defaults(head_nt_constraint=True)

args = parser.parse_args(args=['-data_type', 'django', '-data', 'data/django.cleaned.dataset.freq5.par_info.refact.space_only.bin',
                               '-model', 'models/model.django_word128_encoder256_rule128_node64.beam15.adam.simple_trans.no_unary_closure.8e39832.run3.best_acc.npz'])
if args.data_type == 'hs':
    args.decode_max_time_step = 350

logging.info('loading dataset [%s]', args.data)
train_data, dev_data, test_data = deserialize_from_file(args.data)

if not args.source_vocab_size:
    args.source_vocab_size = train_data.annot_vocab.size
if not args.target_vocab_size:
    args.target_vocab_size = train_data.terminal_vocab.size
if not args.rule_num:
    args.rule_num = len(train_data.grammar.rules)
if not args.node_num:
    args.node_num = len(train_data.grammar.node_type_to_id)

config_module = sys.modules['config']
for name, value in vars(args).iteritems():
    setattr(config_module, name, value)

# build the model
model = Model()
model.build()
model.load(args.model)

def decode_query(query):
    """decode a given natural language query, return a list of generated candidates"""
    query, str_map = canonicalize_query(query)
    vocab = train_data.annot_vocab
    query_tokens = query.split(' ')
    query_tokens_data = [query_to_data(query, vocab)]
    example = namedtuple('example', ['query', 'data'])(query=query_tokens, data=query_tokens_data)

    cand_list = model.decode(example, train_data.grammar, train_data.terminal_vocab,
                             beam_size=args.beam_size, max_time_step=args.decode_max_time_step, log=True)

    return cand_list

if __name__ == '__main__':
    print 'run in interactive mode'
    while True:
        query = raw_input('input a query: ')
        cand_list = decode_query(query)

        # output top 5 candidates
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