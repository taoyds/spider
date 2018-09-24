# -*- coding: UTF-8 -*-

from __future__ import division
import os
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import logging
import traceback

from nn.utils.generic_utils import init_logging

from model import *


DJANGO_ANNOT_FILE = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno'


def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]

    return tokens


def evaluate(model, dataset, verbose=True):
    if verbose:
        logging.info('evaluating [%s] dataset, [%d] examples' % (dataset.name, dataset.count))

    exact_match_ratio = 0.0

    for example in dataset.examples:
        logging.info('evaluating example [%d]' % example.eid)
        hyps, hyp_scores = model.decode(example, max_time_step=config.decode_max_time_step)
        gold_rules = example.rules

        if len(hyps) == 0:
            logging.warning('no decoding result for example [%d]!' % example.eid)
            continue

        best_hyp = hyps[0]
        predict_rules = [dataset.grammar.id_to_rule[rid] for rid in best_hyp]

        assert len(predict_rules) > 0 and len(gold_rules) > 0

        exact_match = sorted(gold_rules, key=lambda x: x.__repr__()) == sorted(predict_rules, key=lambda x: x.__repr__())
        if exact_match:
            exact_match_ratio += 1

        # p = len(predict_rules.intersection(gold_rules)) / len(predict_rules)
        # r = len(predict_rules.intersection(gold_rules)) / len(gold_rules)

    exact_match_ratio /= dataset.count

    logging.info('exact_match_ratio = %f' % exact_match_ratio)

    return exact_match_ratio


def evaluate_decode_results(dataset, decode_results, verbose=True):
    from lang.py.parse import tokenize_code, de_canonicalize_code
    # tokenize_code = tokenize_for_bleu_eval
    import ast
    assert dataset.count == len(decode_results)

    f = f_decode = None
    if verbose:
        f = open(dataset.name + '.exact_match', 'w')
        exact_match_ids = []
        f_decode = open(dataset.name + '.decode_results.txt', 'w')
        eid_to_annot = dict()

        if config.data_type == 'django':
            for raw_id, line in enumerate(open(DJANGO_ANNOT_FILE)):
                eid_to_annot[raw_id] = line.strip()

        f_bleu_eval_ref = open(dataset.name + '.ref', 'w')
        f_bleu_eval_hyp = open(dataset.name + '.hyp', 'w')
        f_generated_code = open(dataset.name + '.geneated_code', 'w')

        logging.info('evaluating [%s] set, [%d] examples', dataset.name, dataset.count)

    cum_oracle_bleu = 0.0
    cum_oracle_acc = 0.0
    cum_bleu = 0.0
    cum_acc = 0.0
    sm = SmoothingFunction()

    all_references = []
    all_predictions = []

    if all(len(cand) == 0 for cand in decode_results):
        logging.ERROR('Empty decoding results for the current dataset!')
        return -1, -1

    for eid in range(dataset.count):
        example = dataset.examples[eid]
        ref_code = example.code
        ref_ast_tree = ast.parse(ref_code).body[0]
        refer_source = astor.to_source(ref_ast_tree).strip()
        # refer_source = ref_code
        refer_tokens = tokenize_code(refer_source)
        cur_example_correct = False

        decode_cands = decode_results[eid]
        if len(decode_cands) == 0:
            continue

        decode_cand = decode_cands[0]

        cid, cand, ast_tree, code = decode_cand
        code = astor.to_source(ast_tree).strip()

        # simple_url_2_re = re.compile('_STR:0_', re.))
        try:
            predict_tokens = tokenize_code(code)
        except:
            logging.error('error in tokenizing [%s]', code)
            continue

        if refer_tokens == predict_tokens:
            cum_acc += 1
            cur_example_correct = True

            if verbose:
                exact_match_ids.append(example.raw_id)
                f.write('-' * 60 + '\n')
                f.write('example_id: %d\n' % example.raw_id)
                f.write(code + '\n')
                f.write('-' * 60 + '\n')

        if config.data_type == 'django':
            ref_code_for_bleu = example.meta_data['raw_code']
            pred_code_for_bleu = de_canonicalize_code(code, example.meta_data['raw_code'])
            # ref_code_for_bleu = de_canonicalize_code(ref_code_for_bleu, example.meta_data['raw_code'])
            # convert canonicalized code to raw code
            for literal, place_holder in example.meta_data['str_map'].iteritems():
                pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
                # ref_code_for_bleu = ref_code_for_bleu.replace('\'' + place_holder + '\'', literal)
        elif config.data_type == 'hs':
            ref_code_for_bleu = ref_code
            pred_code_for_bleu = code

        # we apply Ling Wang's trick when evaluating BLEU scores
        refer_tokens_for_bleu = tokenize_for_bleu_eval(ref_code_for_bleu)
        pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

        # The if-chunk below is for debugging purpose, sometimes the reference cannot match with the prediction
        # because of inconsistent quotes (e.g., single quotes in reference, double quotes in prediction).
        # However most of these cases are solved by cannonicalizing the reference code using astor (parse the reference
        # into AST, and regenerate the code. Use this regenerated one as the reference)
        weired = False
        if refer_tokens_for_bleu == pred_tokens_for_bleu and refer_tokens != predict_tokens:
            # cum_acc += 1
            weired = True
        elif refer_tokens == predict_tokens:
            # weired!
            # weired = True
            pass

        shorter = len(pred_tokens_for_bleu) < len(refer_tokens_for_bleu)

        all_references.append([refer_tokens_for_bleu])
        all_predictions.append(pred_tokens_for_bleu)

        # try:
        ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
        bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu, weights=ngram_weights, smoothing_function=sm.method3)
        cum_bleu += bleu_score
        # except:
        #    pass

        if verbose:
            print 'raw_id: %d, bleu_score: %f' % (example.raw_id, bleu_score)

            f_decode.write('-' * 60 + '\n')
            f_decode.write('example_id: %d\n' % example.raw_id)
            f_decode.write('intent: \n')

            if config.data_type == 'django':
                f_decode.write(eid_to_annot[example.raw_id] + '\n')
            elif config.data_type == 'hs':
                f_decode.write(' '.join(example.query) + '\n')

            f_bleu_eval_ref.write(' '.join(refer_tokens_for_bleu) + '\n')
            f_bleu_eval_hyp.write(' '.join(pred_tokens_for_bleu) + '\n')

            f_decode.write('canonicalized reference: \n')
            f_decode.write(refer_source + '\n')
            f_decode.write('canonicalized prediction: \n')
            f_decode.write(code + '\n')
            f_decode.write('reference code for bleu calculation: \n')
            f_decode.write(ref_code_for_bleu + '\n')
            f_decode.write('predicted code for bleu calculation: \n')
            f_decode.write(pred_code_for_bleu + '\n')
            f_decode.write('pred_shorter_than_ref: %s\n' % shorter)
            f_decode.write('weired: %s\n' % weired)
            f_decode.write('-' * 60 + '\n')

            # for Hiro's evaluation
            f_generated_code.write(pred_code_for_bleu.replace('\n', '#NEWLINE#') + '\n')


        # compute oracle
        best_score = 0.
        cur_oracle_acc = 0.
        for decode_cand in decode_cands[:config.beam_size]:
            cid, cand, ast_tree, code = decode_cand
            try:
                code = astor.to_source(ast_tree).strip()
                predict_tokens = tokenize_code(code)

                if predict_tokens == refer_tokens:
                    cur_oracle_acc = 1

                if config.data_type == 'django':
                    pred_code_for_bleu = de_canonicalize_code(code, example.meta_data['raw_code'])
                    # convert canonicalized code to raw code
                    for literal, place_holder in example.meta_data['str_map'].iteritems():
                        pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
                elif config.data_type == 'hs':
                    pred_code_for_bleu = code

                # we apply Ling Wang's trick when evaluating BLEU scores
                pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

                ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
                bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu,
                                           weights=ngram_weights,
                                           smoothing_function=sm.method3)

                if bleu_score > best_score:
                    best_score = bleu_score

            except:
                continue

        cum_oracle_bleu += best_score
        cum_oracle_acc += cur_oracle_acc

    cum_bleu /= dataset.count
    cum_acc /= dataset.count
    cum_oracle_bleu /= dataset.count
    cum_oracle_acc /= dataset.count

    logging.info('corpus level bleu: %f', corpus_bleu(all_references, all_predictions, smoothing_function=sm.method3))
    logging.info('sentence level bleu: %f', cum_bleu)
    logging.info('accuracy: %f', cum_acc)
    logging.info('oracle bleu: %f', cum_oracle_bleu)
    logging.info('oracle accuracy: %f', cum_oracle_acc)

    if verbose:
        f.write(', '.join(str(i) for i in exact_match_ids))
        f.close()
        f_decode.close()

        f_bleu_eval_ref.close()
        f_bleu_eval_hyp.close()
        f_generated_code.close()

    return cum_bleu, cum_acc


def analyze_decode_results(dataset, decode_results, verbose=True):
    from lang.py.parse import tokenize_code, de_canonicalize_code
    # tokenize_code = tokenize_for_bleu_eval
    import ast
    assert dataset.count == len(decode_results)

    f = f_decode = None
    if verbose:
        f = open(dataset.name + '.exact_match', 'w')
        exact_match_ids = []
        f_decode = open(dataset.name + '.decode_results.txt', 'w')
        eid_to_annot = dict()

        if config.data_type == 'django':
            for raw_id, line in enumerate(open(DJANGO_ANNOT_FILE)):
                eid_to_annot[raw_id] = line.strip()

        f_bleu_eval_ref = open(dataset.name + '.ref', 'w')
        f_bleu_eval_hyp = open(dataset.name + '.hyp', 'w')

        logging.info('evaluating [%s] set, [%d] examples', dataset.name, dataset.count)

    cum_oracle_bleu = 0.0
    cum_oracle_acc = 0.0
    cum_bleu = 0.0
    cum_acc = 0.0
    sm = SmoothingFunction()

    all_references = []
    all_predictions = []

    if all(len(cand) == 0 for cand in decode_results):
        logging.ERROR('Empty decoding results for the current dataset!')
        return -1, -1

    binned_results_dict = defaultdict(list)
    def get_binned_key(ast_size):
        cutoff = 50 if config.data_type == 'django' else 250
        k = 10 if config.data_type == 'django' else 25 # for hs

        if ast_size >= cutoff:
            return '%d - inf' % cutoff

        lower = int(ast_size / k) * k
        upper = lower + k

        key = '%d - %d' % (lower, upper)

        return key


    for eid in range(dataset.count):
        example = dataset.examples[eid]
        ref_code = example.code
        ref_ast_tree = ast.parse(ref_code).body[0]
        refer_source = astor.to_source(ref_ast_tree).strip()
        # refer_source = ref_code
        refer_tokens = tokenize_code(refer_source)
        cur_example_acc = 0.0

        decode_cands = decode_results[eid]
        if len(decode_cands) == 0:
            continue

        decode_cand = decode_cands[0]

        cid, cand, ast_tree, code = decode_cand
        code = astor.to_source(ast_tree).strip()

        # simple_url_2_re = re.compile('_STR:0_', re.))
        try:
            predict_tokens = tokenize_code(code)
        except:
            logging.error('error in tokenizing [%s]', code)
            continue

        if refer_tokens == predict_tokens:
            cum_acc += 1
            cur_example_acc = 1.0

            if verbose:
                exact_match_ids.append(example.raw_id)
                f.write('-' * 60 + '\n')
                f.write('example_id: %d\n' % example.raw_id)
                f.write(code + '\n')
                f.write('-' * 60 + '\n')

        if config.data_type == 'django':
            ref_code_for_bleu = example.meta_data['raw_code']
            pred_code_for_bleu = de_canonicalize_code(code, example.meta_data['raw_code'])
            # ref_code_for_bleu = de_canonicalize_code(ref_code_for_bleu, example.meta_data['raw_code'])
            # convert canonicalized code to raw code
            for literal, place_holder in example.meta_data['str_map'].iteritems():
                pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
                # ref_code_for_bleu = ref_code_for_bleu.replace('\'' + place_holder + '\'', literal)
        elif config.data_type == 'hs':
            ref_code_for_bleu = ref_code
            pred_code_for_bleu = code

        # we apply Ling Wang's trick when evaluating BLEU scores
        refer_tokens_for_bleu = tokenize_for_bleu_eval(ref_code_for_bleu)
        pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

        shorter = len(pred_tokens_for_bleu) < len(refer_tokens_for_bleu)

        all_references.append([refer_tokens_for_bleu])
        all_predictions.append(pred_tokens_for_bleu)

        # try:
        ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
        bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu, weights=ngram_weights, smoothing_function=sm.method3)
        cum_bleu += bleu_score
        # except:
        #    pass

        if verbose:
            print 'raw_id: %d, bleu_score: %f' % (example.raw_id, bleu_score)

            f_decode.write('-' * 60 + '\n')
            f_decode.write('example_id: %d\n' % example.raw_id)
            f_decode.write('intent: \n')

            if config.data_type == 'django':
                f_decode.write(eid_to_annot[example.raw_id] + '\n')
            elif config.data_type == 'hs':
                f_decode.write(' '.join(example.query) + '\n')

            f_bleu_eval_ref.write(' '.join(refer_tokens_for_bleu) + '\n')
            f_bleu_eval_hyp.write(' '.join(pred_tokens_for_bleu) + '\n')

            f_decode.write('canonicalized reference: \n')
            f_decode.write(refer_source + '\n')
            f_decode.write('canonicalized prediction: \n')
            f_decode.write(code + '\n')
            f_decode.write('reference code for bleu calculation: \n')
            f_decode.write(ref_code_for_bleu + '\n')
            f_decode.write('predicted code for bleu calculation: \n')
            f_decode.write(pred_code_for_bleu + '\n')
            f_decode.write('pred_shorter_than_ref: %s\n' % shorter)
            # f_decode.write('weired: %s\n' % weired)
            f_decode.write('-' * 60 + '\n')

        # compute oracle
        best_bleu_score = 0.
        cur_oracle_acc = 0.
        for decode_cand in decode_cands[:config.beam_size]:
            cid, cand, ast_tree, code = decode_cand
            try:
                code = astor.to_source(ast_tree).strip()
                predict_tokens = tokenize_code(code)

                if predict_tokens == refer_tokens:
                    cur_oracle_acc = 1.

                if config.data_type == 'django':
                    pred_code_for_bleu = de_canonicalize_code(code, example.meta_data['raw_code'])
                    # convert canonicalized code to raw code
                    for literal, place_holder in example.meta_data['str_map'].iteritems():
                        pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
                elif config.data_type == 'hs':
                    pred_code_for_bleu = code

                # we apply Ling Wang's trick when evaluating BLEU scores
                pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

                ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
                cand_bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu,
                                                weights=ngram_weights,
                                                smoothing_function=sm.method3)

                if cand_bleu_score > best_bleu_score:
                    best_bleu_score = cand_bleu_score

            except:
                continue

        cum_oracle_bleu += best_bleu_score
        cum_oracle_acc += cur_oracle_acc

        ref_ast_size = example.parse_tree.size
        binned_key = get_binned_key(ref_ast_size)
        binned_results_dict[binned_key].append((bleu_score, cur_example_acc, best_bleu_score, cur_oracle_acc))

    cum_bleu /= dataset.count
    cum_acc /= dataset.count
    cum_oracle_bleu /= dataset.count
    cum_oracle_acc /= dataset.count

    logging.info('corpus level bleu: %f', corpus_bleu(all_references, all_predictions, smoothing_function=sm.method3))
    logging.info('sentence level bleu: %f', cum_bleu)
    logging.info('accuracy: %f', cum_acc)
    logging.info('oracle bleu: %f', cum_oracle_bleu)
    logging.info('oracle accuracy: %f', cum_oracle_acc)

    keys = sorted(binned_results_dict, key=lambda x: int(x.split(' - ')[0]))

    Y = [[], [], [], []]
    X = []

    for binned_key in keys:
        entry = binned_results_dict[binned_key]
        avg_bleu = np.average([t[0] for t in entry])
        avg_acc = np.average([t[1] for t in entry])
        avg_oracle_bleu = np.average([t[2] for t in entry])
        avg_oracle_acc = np.average([t[3] for t in entry])
        print binned_key, avg_bleu, avg_acc, avg_oracle_bleu, avg_oracle_acc, len(entry)

        Y[0].append(avg_bleu)
        Y[1].append(avg_acc)
        Y[2].append(avg_oracle_bleu)
        Y[3].append(avg_oracle_acc)

        X.append(int(binned_key.split(' - ')[0]))

    import matplotlib.pyplot as plt
    from pylab import rcParams
    rcParams['figure.figsize'] = 6, 2.5

    if config.data_type == 'django':
        fig, ax = plt.subplots()
        ax.plot(X, Y[0], 'bs--', label='BLEU', lw=1.2)
        # ax.plot(X, Y[2], 'r^--', label='oracle BLEU', lw=1.2)
        ax.plot(X, Y[1], 'r^--', label='acc', lw=1.2)
        # ax.plot(X, Y[3], 'r^--', label='oracle acc', lw=1.2)
        ax.set_ylabel('Performance')
        ax.set_xlabel('Reference AST Size (# nodes)')
        plt.legend(loc='upper right', ncol=6)
        plt.tight_layout()
        # plt.savefig('django_acc_ast_size.pdf', dpi=300)
        # os.system('pcrop.sh django_acc_ast_size.pdf')
        plt.savefig('django_perf_ast_size.pdf', dpi=300)
        os.system('pcrop.sh django_perf_ast_size.pdf')
    else:
        fig, ax = plt.subplots()
        ax.plot(X, Y[0], 'bs--', label='BLEU', lw=1.2)
        # ax.plot(X, Y[2], 'r^--', label='oracle BLEU', lw=1.2)
        ax.plot(X, Y[1], 'r^--', label='acc', lw=1.2)
        # ax.plot(X, Y[3], 'r^--', label='oracle acc', lw=1.2)
        ax.set_ylabel('Performance')
        ax.set_xlabel('Reference AST Size (# nodes)')
        plt.legend(loc='upper right', ncol=6)
        plt.tight_layout()
        # plt.savefig('hs_bleu_ast_size.pdf', dpi=300)
        # os.system('pcrop.sh hs_bleu_ast_size.pdf')
        plt.savefig('hs_perf_ast_size.pdf', dpi=300)
        os.system('pcrop.sh hs_perf_ast_size.pdf')
    if verbose:
        f.write(', '.join(str(i) for i in exact_match_ids))
        f.close()
        f_decode.close()

        f_bleu_eval_ref.close()
        f_bleu_eval_hyp.close()

    return cum_bleu, cum_acc


def evaluate_seq2seq_decode_results(dataset, seq2seq_decode_file, seq2seq_ref_file, verbose=True, is_nbest=False):
    from lang.py.parse import parse

    f_seq2seq_decode = open(seq2seq_decode_file)
    f_seq2seq_ref = open(seq2seq_ref_file)

    if verbose:
        logging.info('evaluating [%s] set, [%d] examples', dataset.name, dataset.count)

    cum_bleu = 0.0
    cum_acc = 0.0
    sm = SmoothingFunction()

    decode_file_data = [l.strip() for l in f_seq2seq_decode.readlines()]
    ref_code_data = [l.strip() for l in f_seq2seq_ref.readlines()]

    if is_nbest:
        for i in xrange(len(decode_file_data)):
            d = decode_file_data[i].split(' ||| ')
            decode_file_data[i] = (int(d[0]), d[1])

    def is_well_formed_python_code(_hyp):
        try:
            _hyp = _hyp.replace('#NEWLINE#', '\n').replace('#INDENT#', '    ').replace(' #MERGE# ', '')
            hyp_ast_tree = parse(_hyp)
            return True
        except:
            return False

    for eid in range(dataset.count):
        example = dataset.examples[eid]
        cur_example_correct = False

        if is_nbest:
            # find the best-scored well-formed code from the n-best list
            n_best_list = filter(lambda x: x[0] == eid, decode_file_data)
            code = top_scored_code = n_best_list[0][1]
            for _, hyp in n_best_list:
                if is_well_formed_python_code(hyp):
                    code = hyp
                    break

            if top_scored_code != code:
                print '*' * 60
                print top_scored_code
                print code
                print '*' * 60

            code = n_best_list[0][1]
        else:
            code = decode_file_data[eid]

        code = code.replace('#NEWLINE#', '\n').replace('#INDENT#', '    ').replace(' #MERGE# ', '')
        ref_code = ref_code_data[eid].replace('#NEWLINE#', '\n').replace('#INDENT#', '    ').replace(' #MERGE# ', '')

        if code == ref_code:
            cum_acc += 1
            cur_example_correct = True


        if config.data_type == 'django':
            ref_code_for_bleu = example.meta_data['raw_code']
            pred_code_for_bleu = code # de_canonicalize_code(code, example.meta_data['raw_code'])
            # ref_code_for_bleu = de_canonicalize_code(ref_code_for_bleu, example.meta_data['raw_code'])
            # convert canonicalized code to raw code
            for literal, place_holder in example.meta_data['str_map'].iteritems():
                pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
                # ref_code_for_bleu = ref_code_for_bleu.replace('\'' + place_holder + '\'', literal)
        elif config.data_type == 'hs':
            ref_code_for_bleu = example.code
            pred_code_for_bleu = code

        # we apply Ling Wang's trick when evaluating BLEU scores
        refer_tokens_for_bleu = tokenize_for_bleu_eval(ref_code_for_bleu)
        pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

        ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
        bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu, weights=ngram_weights, smoothing_function=sm.method3)
        cum_bleu += bleu_score

    cum_bleu /= dataset.count
    cum_acc /= dataset.count

    logging.info('sentence level bleu: %f', cum_bleu)
    logging.info('accuracy: %f', cum_acc)


def evaluate_seq2tree_sample_file(sample_file, id_file, dataset):
    from lang.py.parse import tokenize_code, de_canonicalize_code
    import ast, astor
    import traceback
    from lang.py.seq2tree_exp import seq2tree_repr_to_ast_tree, merge_broken_value_nodes
    from lang.py.parse import decode_tree_to_python_ast

    f_sample = open(sample_file)
    line_id_to_raw_id = OrderedDict()
    raw_id_to_eid = OrderedDict()
    for i, line in enumerate(open(id_file)):
        raw_id = int(line.strip())
        line_id_to_raw_id[i] = raw_id

    for eid in range(len(dataset.examples)):
        raw_id_to_eid[dataset.examples[eid].raw_id] = eid

    rare_word_map = defaultdict(dict)
    if config.seq2tree_rareword_map:
        logging.info('use rare word map')
        for i, line in enumerate(open(config.seq2tree_rareword_map)):
            line = line.strip()
            if line:
                for e in line.split(' '):
                    d = e.split(':', 1)
                    rare_word_map[i][int(d[0])] = d[1]

    cum_bleu = 0.0
    cum_acc = 0.0
    sm = SmoothingFunction()
    convert_error_num = 0

    for i in range(len(line_id_to_raw_id)):
        # print 'working on %d' % i
        ref_repr = f_sample.readline().strip()
        predict_repr = f_sample.readline().strip()
        predict_repr = predict_repr.replace('<U>', 'str{}{unk}') # .replace('( )', '( str{}{unk} )')
        f_sample.readline()

        # if ' ( ) ' in ref_repr:
        #     print i, ref_repr

        if i in rare_word_map:
            for unk_id, w in rare_word_map[i].iteritems():
                ref_repr = ref_repr.replace(' str{}{unk_%s} ' % unk_id, ' str{}{%s} ' % w)
                predict_repr = predict_repr.replace(' str{}{unk_%s} ' % unk_id, ' str{}{%s} ' % w)

        try:
            parse_tree = seq2tree_repr_to_ast_tree(predict_repr)
            merge_broken_value_nodes(parse_tree)
        except:
            print 'error when converting:'
            print predict_repr
            convert_error_num += 1
            continue

        raw_id = line_id_to_raw_id[i]
        eid = raw_id_to_eid[raw_id]
        example = dataset.examples[eid]

        ref_code = example.code
        ref_ast_tree = ast.parse(ref_code).body[0]
        refer_source = astor.to_source(ref_ast_tree).strip()
        refer_tokens = tokenize_code(refer_source)

        try:
            ast_tree = decode_tree_to_python_ast(parse_tree)
            code = astor.to_source(ast_tree).strip()
        except:
            print "Exception in converting tree to code:"
            print '-' * 60
            print 'line id: %d' % i
            traceback.print_exc(file=sys.stdout)
            print '-' * 60
            convert_error_num += 1
            continue

        if config.data_type == 'django':
            ref_code_for_bleu = example.meta_data['raw_code']
            pred_code_for_bleu = de_canonicalize_code(code, example.meta_data['raw_code'])
            # convert canonicalized code to raw code
            for literal, place_holder in example.meta_data['str_map'].iteritems():
                pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
        elif config.data_type == 'hs':
            ref_code_for_bleu = ref_code
            pred_code_for_bleu = code

        # we apply Ling Wang's trick when evaluating BLEU scores
        refer_tokens_for_bleu = tokenize_for_bleu_eval(ref_code_for_bleu)
        pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

        predict_tokens = tokenize_code(code)
        # if ref_repr == predict_repr:
        if predict_tokens == refer_tokens:
            cum_acc += 1

        ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
        bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu, weights=ngram_weights,
                                   smoothing_function=sm.method3)
        cum_bleu += bleu_score

    cum_bleu /= len(line_id_to_raw_id)
    cum_acc /= len(line_id_to_raw_id)
    logging.info('nun. examples: %d', len(line_id_to_raw_id))
    logging.info('num. errors when converting repr to tree: %d', convert_error_num)
    logging.info('ratio of grammatically incorrect trees: %f', convert_error_num / float(len(line_id_to_raw_id)))
    logging.info('sentence level bleu: %f', cum_bleu)
    logging.info('accuracy: %f', cum_acc)


def evaluate_ifttt_results(dataset, decode_results, verbose=True):
    assert dataset.count == len(decode_results)

    f = f_decode = None
    if verbose:
        f = open(dataset.name + '.exact_match', 'w')
        exact_match_ids = []
        f_decode = open(os.path.join(config.output_dir, dataset.name + '.decode_results.txt'), 'w')

        logging.info('evaluating [%s] set, [%d] examples', dataset.name, dataset.count)

    cum_channel_acc = 0.0
    cum_channel_func_acc = 0.0
    cum_prod_f1 = 0.0
    cum_oracle_prod_f1 = 0.0

    if all(len(cand) == 0 for cand in decode_results):
        logging.ERROR('Empty decoding results for the current dataset!')
        return -1, -1, -1

    for eid in range(dataset.count):
        example = dataset.examples[eid]
        ref_parse_tree = example.parse_tree
        decode_candidates = decode_results[eid]

        if len(decode_candidates) == 0:
            continue

        decode_cand = decode_candidates[0]

        cid, cand_hyp = decode_cand
        predict_parse_tree = cand_hyp.tree

        exact_match = predict_parse_tree == ref_parse_tree

        channel_acc, channel_func_acc, prod_f1 = ifttt_metric(predict_parse_tree, ref_parse_tree)
        cum_channel_acc += channel_acc
        cum_channel_func_acc += channel_func_acc
        cum_prod_f1 += prod_f1

        if verbose:
            if exact_match:
                exact_match_ids.append(example.raw_id)

            print 'raw_id: %d, prod_f1: %f' % (example.raw_id, prod_f1)

            f_decode.write('-' * 60 + '\n')
            f_decode.write('example_id: %d\n' % example.raw_id)
            f_decode.write('intent: \n')

            f_decode.write(' '.join(example.query) + '\n')

            f_decode.write('reference: \n')
            f_decode.write(str(ref_parse_tree) + '\n')
            f_decode.write('prediction: \n')
            f_decode.write(str(predict_parse_tree) + '\n')
            f_decode.write('-' * 60 + '\n')

        # compute oracle
        best_prod_f1 = -1.
        for decode_cand in decode_candidates[:10]:
            cid, cand_hyp = decode_cand
            predict_parse_tree = cand_hyp.tree

            channel_acc, channel_func_acc, prod_f1 = ifttt_metric(predict_parse_tree, ref_parse_tree)

            if prod_f1 > best_prod_f1:
                best_prod_f1 = prod_f1

        cum_oracle_prod_f1 += best_prod_f1

    cum_channel_acc /= dataset.count
    cum_channel_func_acc /= dataset.count
    cum_prod_f1 /= dataset.count
    cum_oracle_prod_f1 /= dataset.count

    logging.info('channel_acc: %f', cum_channel_acc)
    logging.info('channel_func_acc: %f', cum_channel_func_acc)
    logging.info('prod_f1: %f', cum_prod_f1)
    logging.info('oracle prod_f1: %f', cum_oracle_prod_f1)

    if verbose:
        f.write(', '.join(str(i) for i in exact_match_ids))
        f.close()
        f_decode.close()

    return cum_channel_acc, cum_channel_func_acc, cum_prod_f1


def ifttt_metric(predict_parse_tree, ref_parse_tree):
    channel_acc = channel_func_acc = prod_f1 = 0.
    # channel acc.
    channel_match = False
    if predict_parse_tree['TRIGGER'].children[0].type == ref_parse_tree['TRIGGER'].children[0].type and \
                    predict_parse_tree['ACTION'].children[0].type == ref_parse_tree['ACTION'].children[0].type:
        channel_acc += 1.
        channel_match = True

    # channel+func acc.
    if channel_match and predict_parse_tree['TRIGGER'].children[0].children[0].type == ref_parse_tree['TRIGGER'].children[0].children[0].type and \
                    predict_parse_tree['ACTION'].children[0].children[0].type == ref_parse_tree['ACTION'].children[0].children[0].type:
        channel_func_acc += 1.

    # predict_parse_tree is of type DecodingTree, different from reference tree!
    # if predict_parse_tree == ref_parse_tree:
    #     channel_func_acc += 1.

    # prod. F1
    ref_rules, _ = ref_parse_tree.get_productions()
    predict_rules, _ = predict_parse_tree.get_productions()

    prod_f1 = len(set(ref_rules).intersection(set(predict_rules))) / len(ref_rules)

    return channel_acc, channel_func_acc, prod_f1


def decode_and_evaluate_ifttt(model, test_data):
    raw_ids = [int(i.strip()) for i in open(config.ifttt_test_split)]  # 'data/ifff.test_data.gold.id'
    eids  = [i for i, e in enumerate(test_data.examples) if e.raw_id in raw_ids]
    test_data_subset = test_data.get_dataset_by_ids(eids, test_data.name + '.subset')

    from decoder import decode_ifttt_dataset
    decode_results = decode_ifttt_dataset(model, test_data_subset, verbose=True)
    evaluate_ifttt_results(test_data_subset, decode_results)

    return decode_results


def decode_and_evaluate_ifttt_by_split(model, test_data):
    for split in ['ifff.test_data.omit_non_english.id', 'ifff.test_data.omit_unintelligible.id', 'ifff.test_data.gold.id']:
        raw_ids = [int(i.strip()) for i in open(os.path.join(config.ifttt_test_split), split)]  # 'data/ifff.test_data.gold.id'
        eids = [i for i, e in enumerate(test_data.examples) if e.raw_id in raw_ids]
        test_data_subset = test_data.get_dataset_by_ids(eids, test_data.name + '.' + split)

        from decoder import decode_ifttt_dataset
        decode_results = decode_ifttt_dataset(model, test_data_subset, verbose=True)
        evaluate_ifttt_results(test_data_subset, decode_results)


if __name__ == '__main__':
    from dataset import DataEntry, DataSet, Vocab, Action
    init_logging('parser.log', logging.INFO)

    train_data, dev_data, test_data = deserialize_from_file('data/ifttt.freq3.bin')
    decoding_results = []
    for eid in range(test_data.count):
        example = test_data.examples[eid]
        decoding_results.append([(eid, example.parse_tree)])

    evaluate_ifttt_results(test_data, decoding_results, verbose=True)
