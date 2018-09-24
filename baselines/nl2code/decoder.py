import traceback
import config
import json
from model import *

def decode_sql_dataset(model, dataset, output_path,verbose=True):
    from lang.sql.parse import decode_tree_to_sql_ast
    if verbose:
        logging.info('decoding [%s] set, num. examples: %d', dataset.name, dataset.count)

    decode_results = []
    cum_num = 0
    for example in dataset.examples:
        cand_list = model.decode(example, dataset.grammar, dataset.terminal_vocab,
                                 beam_size=config.beam_size, max_time_step=config.decode_max_time_step)

        exg_decode_results = []
        for cid, cand in enumerate(cand_list[:10]):
            try:
                ast_tree = decode_tree_to_sql_ast(cand.tree)
                # print(ast_tree)
                # code = astor.to_source(ast_tree)
                if cid == 0:
                    exg_decode_results.append((cid, ast_tree))
            except:
                if verbose:
                    print "Exception in converting tree to code:"
                    print '-' * 60
                    print 'raw_id: %d, beam pos: %d' % (example.raw_id, cid)
                    traceback.print_exc(file=sys.stdout)
                    print '-' * 60

        cum_num += 1
        if cum_num % 50 == 0 and verbose:
            print '%d examples so far ...' % cum_num

        decode_results.append(exg_decode_results)
        f = open(output_path,"w")
        json.dump(decode_results,f)
        f.close()
    return decode_results

def decode_python_dataset(model, dataset, verbose=True):
    from lang.sql.parse import decode_tree_to_python_ast
    if verbose:
        logging.info('decoding [%s] set, num. examples: %d', dataset.name, dataset.count)

    decode_results = []
    cum_num = 0
    for example in dataset.examples:
        cand_list = model.decode(example, dataset.grammar, dataset.terminal_vocab,
                                 beam_size=config.beam_size, max_time_step=config.decode_max_time_step)

        exg_decode_results = []
        for cid, cand in enumerate(cand_list[:10]):
            try:
                ast_tree = decode_tree_to_python_ast(cand.tree)
                code = astor.to_source(ast_tree)
                exg_decode_results.append((cid, cand, ast_tree, code))
            except:
                if verbose:
                    print "Exception in converting tree to code:"
                    print '-' * 60
                    print 'raw_id: %d, beam pos: %d' % (example.raw_id, cid)
                    traceback.print_exc(file=sys.stdout)
                    print '-' * 60

        cum_num += 1
        if cum_num % 50 == 0 and verbose:
            print '%d examples so far ...' % cum_num

        decode_results.append(exg_decode_results)

    return decode_results

    # serialize_to_file(decode_results, '%s.decode_results.profile' % dataset.name)

def decode_ifttt_dataset(model, dataset, verbose=True):
    if verbose:
        logging.info('decoding [%s] set, num. examples: %d', dataset.name, dataset.count)

    decode_results = []
    cum_num = 0
    for example in dataset.examples:
        cand_list = model.decode(example, dataset.grammar, dataset.terminal_vocab,
                                 beam_size=config.beam_size, max_time_step=config.decode_max_time_step)

        exg_decode_results = []
        for cid, cand in enumerate(cand_list[:10]):
            try:
                exg_decode_results.append((cid, cand))
            except:
                if verbose:
                    print "Exception in converting tree to code:"
                    print '-' * 60
                    print 'raw_id: %d, beam pos: %d' % (example.raw_id, cid)
                    traceback.print_exc(file=sys.stdout)
                    print '-' * 60

        cum_num += 1
        if cum_num % 50 == 0 and verbose:
            print '%d examples so far ...' % cum_num

        decode_results.append(exg_decode_results)

    return decode_results
