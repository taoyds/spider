"""
Script to simplify running experiments. Given a simplified
specification file, generates the train.yml, infer.yml, and
infer_train.yml files with the detailed specifications. These files
will incorporate default values for rarely changed settings, as well
as generalizing some information (e.g., if we are using a copying
input pipeline, we should use the corresponding copying model and
copying decoder). Also generates a bash script that will run the
experiment and run quick_eval on the results.
"""

import argparse
import os
import sys
import yaml

from collections import Counter

# todo: set the maximum length, see the generate_vocab.py
def ensure_in_spec(spec, key):
    """
    If key is not in spec, requests a value from the user and modifies
    spec to include that key:value pair.
    Args:
      spec: A dictionary specifying the experiment.
      key: A string that may or may not be in the dictionary.
    Modifies:
      spec
    """
    if not key in spec:
        value = raw_input("Please enter a value for %s, or q to quit: " % key)
        if value.lower() == "q":
            print "Quitting."
            sys.exit(0)
        spec[key] = value

def parse_yaml(spec_file):
    print("Parsing simple specification. (not fully implemented)")
    with open(spec_file, "r") as f:
        spec = yaml.load(f)
    return spec

def expand_spec(spec):
    print("Expanding specification. (not fully implemented)")
    ensure_in_spec(spec, "model_dir")
    spec["model_dir"] = os.path.expanduser(spec["model_dir"])
    ensure_in_spec(spec, "data_directories")
    ensure_in_spec(spec, "model")
    expand_model(spec)
    expand_data_dirs(spec)

    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(spec)

    return spec

def expand_model(spec):
    model = spec["model"]
    # If it's a copying model, set the decode vocab.
    vocab_fnames = {
        "SchemaAttentionCopyingSeq2Seq" : "decode_copy_schema_vocab.txt",
        "InputAttentionCopyingSeq2Seq" : "decode_copy_encode_vocab.txt",
        "SchemaAndInputAttentionCopyingSeq2Seq" : "decode_copy_both_vocab.txt"
    }
    if model in vocab_fnames:
        spec["decode_vocab"] = vocab_fnames[model]

    # Use the model to choose the input pipeline.
    input_pipelines = {
        "BasicSeq2Seq" :
          # "ParallelTextInputPipeline",
        "ParallelTextAndMaskInputPipeline",
        "AttentionSeq2Seq" :
          # "ParallelTextInputPipeline",
         "ParallelTextAndMaskInputPipeline",
        "InputAttentionCopyingSeq2Seq":
          # "ParallelTextCopyingPipeline"
        # "ParallelTextAndMaskInputPipeline"
        "ParallelTextAndMaskCopyingPipeline"

    }
    spec["input_pipeline"] = input_pipelines[model]

def expand_data_dirs(spec):
    """
    Makes sure all data directories exist and contain required
    files. Then adds the locations of train_encode, train_decode,
    train_schema_locations, dev_encode, dev_decode,
    dev_schema_locations, and tentative locations of vocabulary to
    spec.
    Args:
      spec: a specification that contains "data_dir"
    Modifies:
      spec
    """
    # Make sure directory exists.
    data_dirs = [spec["data_directories"]]
    required_files = {
        "train/train_encode.txt":[],
        #"train/train_schema_locations.txt":[],
        "dev/dev_encode.txt":[],
        #"dev/dev_schema_locations.txt":[],
        "test/test_encode.txt":[],
        #"test/test_schema_locations.txt":[],
        "encode_vocab.txt":[],
        "train/train_decoder_mask.txt":[],
        "dev/dev_decoder_mask.txt":[],
        "test/test_decoder_mask.txt":[]
    }

    non_schema_copying = {
        "train/train_decode.txt":[],
        "dev/dev_decode.txt":[],
        "test/test_decode.txt":[]
    }

    schema_copying = {
        "train/train_copy_decode.txt":[],
        "dev/dev_copy_decode.txt":[],
        "test/test_copy_decode.txt":[]
    }

    # If it's a schema copying or both model, have it use the preprocessed
    # input files.
    if spec["model"] in {"SchemaAttentionCopyingSeq2Seq",
                         "SchemaAndInputAttentionCopyingSeq2Seq"}:
        required_files.update(schema_copying)
    else:
        required_files.update(non_schema_copying)

    # We might need a specific decode vocab if we're using
    # copying models. If so, it will be in the spec.
    decode_vocab_fname = "decode_vocab.txt"
    if "decode_vocab" in spec:
        decode_vocab_fname = spec["decode_vocab"]
    required_files[decode_vocab_fname] = []

    for data_dir in data_dirs:
        data_dir = os.path.expanduser(data_dir)
        if not os.path.isdir(data_dir):
            raise ValueError("Data directory %s does not exist." % data_dir)

        # Make sure directory contains required files.
        for fname in required_files.keys():
            path = os.path.join(data_dir, fname)
            if not os.path.isfile(path):
                raise ValueError("Required data file %s does not "
                                 " exist." % path)
            required_files[fname].append(path)

    # We use only one encode and one decode vocabulary. If we have
    # more than one, we combine them into one  and put it in our model
    # directory.
    required_files["encode_vocab.txt"] = single_vocab_file(
        spec, required_files, "encode_vocab.txt")
    required_files[decode_vocab_fname] = single_vocab_file(
        spec, required_files, decode_vocab_fname)

    # Make it easy to find the decode_vocab later
    required_files["decode_vocab"] = required_files[decode_vocab_fname]

    spec["required_files"] = required_files

def single_vocab_file(spec, required_files_dict, vocab_fname):
    """
    Given a list of vocabulary file paths, return a path to a single
    vocabulary file that incorporates all of the vocabularies. If the
    list is a single path, this simply returns that path. Otherwise,
    we build a combined vocabulary file in the model_dir from spec
    and return the path to that file.
    """
    vocab_file_list = required_files_dict[vocab_fname]
    if len(vocab_file_list) == 1:
        return vocab_file_list[0]
    vocab = Counter()
    for fname in vocab_file_list:
        print fname
        vocab = update_vocab(vocab, fname)
        print "vocab size is now %d" % len(vocab)

    vocab_string = ""
    for word, count in vocab.most_common():
        vocab_string += "%s\t%d\n" %(word, count)

    model_dir = spec["model_dir"]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    output_location = os.path.join(model_dir, vocab_fname)
    with open(output_location, 'w') as f:
        f.write(vocab_string)
    return output_location

def update_vocab(vocab, fname):
    with open(fname, 'r') as f:
        lines = [l.strip() for l in f.readlines() if len(l.strip()) > 0]
    wordcount = Counter()
    for l in lines:
        l = l.split("\t")
        if not len(l) == 2: continue
        word = l[0]
        count = int(l[1])
        wordcount[word] = count
    vocab += wordcount
    return vocab

def generate_training_spec(spec, include_dev=False):
    training_spec = {}
    if "hooks" in spec:
        training_spec["hooks"] = spec["hooks"]
    # else:
    #     training_spec["hooks"] = [{'class': 'PrintModelAnalysisHook'},
    #                               {'class': 'MetadataCaptureHook'},
    #                               {'class': 'SyncReplicasOptimizerHook'},
    #                               {'class': 'TrainSampleHook',
                                   # 'params': {'every_n_steps': 1000}}]
    training_spec["model"] = spec["model"]

    model_params = {}
    if "model_params" in spec:
        model_params = spec["model_params"]
    if not "vocab_source" in model_params:
        model_params["vocab_source"] = spec["required_files"]["encode_vocab.txt"]
    if not "vocab_target" in model_params:
        model_params["vocab_target"] = spec["required_files"]["decode_vocab"]
    training_spec["model_params"] = model_params

    train_decode_fname = "train/train_decode.txt"
    dev_decode_fname = "dev/dev_decode.txt"
    # If it's a schema copying or both model, have it use the preprocessed
    # input files.
    if spec["model"] in {"SchemaAttentionCopyingSeq2Seq",
                 "SchemaAndInputAttentionCopyingSeq2Seq"}:
        train_decode_fname = "train/train_copy_decode.txt"
        dev_decode_fname = "dev/dev_copy_decode.txt"

    if include_dev:
        make_input_pipeline("input_pipeline_train", spec, training_spec,
                            ["train/train_encode.txt", "dev/dev_encode.txt"],
                            # [#"train/train_schema_locations.txt",
                            #  #"dev/dev_schema_locations.txt"
                            # ],
                            None,
                            [train_decode_fname, dev_decode_fname],
                           ["train/train_decoder_mask.txt", "dev/dev_decoder_mask.txt"])
    else:
        make_input_pipeline("input_pipeline_train", spec, training_spec,
                            "train/train_encode.txt",
                            #"train/train_schema_locations.txt",
                            None,
                            train_decode_fname,
                            "train/train_decoder_mask.txt"
                           )
    make_input_pipeline("input_pipeline_dev", spec, training_spec,
                        "dev/dev_encode.txt",
                        #"dev/dev_schema_locations.txt",
                        None,
                        dev_decode_fname,
                       "dev/dev_decoder_mask.txt")
    # buckets?
    if "buckets" in spec:
        training_spec["buckets"] = spec["buckets"]
    else:
        training_spec["buckets"] = "25,50,100,200"

    if "batch_size" in spec:
        training_spec["batch_size"] = spec["batch_size"]
    else:
        training_spec["batch_size"] = 16

    training_spec["output_dir"] = spec["model_dir"]

    if "train_steps" in spec:
        training_spec["train_steps"] = spec["train_steps"]
    else:
        training_spec["train_steps"] = 40000

    if "save_checkpoints_steps" in spec:
        training_spec["save_checkpoints_steps"] = spec["save_checkpoints_steps"]
    else:
        training_spec["save_checkpoints_steps"] = 2000

    if "keep_checkpoint_max" in spec:
        training_spec["keep_checkpoint_max"] = spec["keep_checkpoint_max"]
    else:
        training_spec["keep_checkpoint_max"] = 0



    # Use defaults from train.py for these, unless we overwrite them
    # in the config file.
    usually_defaults = ["tf_random_seed", "save_checkpoints_secs",
                        "schedule", "eval_every_n_steps",
                        "keep_checkpoint_every_n_hours",
                        "gpu_memory_fraction",
                        "gpu_allow_growth", "log_device_placement"]
    for param in usually_defaults:
        if param in spec:
            training_spec[param] = spec[param]

    print "==========================="
    print "Training Spec:"
    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(training_spec)

    return training_spec

def make_input_pipeline(pipeline_name, input_spec, output_spec,
                        source_fname, schema_fname=None, target_fname=None, mask_fname=None):
    """
    Builds the input pipeline configuration and adds it to the
    output_spec.
    Args:
      pipeline_name: the name of the pipeline to add to the
                     output_spec. E.g., "input_pipeline_train"
      input_spec: the working specification (usually spec)
      output_spec: the spec you're building, to which the new pipeline
                   dict will be added. E.g., training_spec
      source_fname: base filename for the source files; what you would
                    use to look them up in
                    input_spec["required_files"]. E.g.,
                    "train/train_encode.txt"
      schema_fname: base filename for the schema locations files.
      target_fname: base filename for the target files, similar to
                    source_fname. May be omitted (e.g., for infer)
    """
    input_pipeline = {}
    if pipeline_name in input_spec and isinstance(input_spec[pipeline_name],
                                                  dict):
        input_pipeline = input_spec[pipeline_name]
    if not "class" in input_pipeline:
        input_pipeline["class"] = input_spec["input_pipeline"]
    params = {}
    if "params" in input_pipeline:
        params = input_pipeline["params"]
    params["source_files"] = build_file_list(source_fname, input_spec)
    if schema_fname:
        # Check if this is a model that uses schemas
        model_name = input_spec["model"]
        if "schema" in model_name.lower() or "copy" in model_name.lower():
            params["schema_loc_files"] = build_file_list(schema_fname, input_spec)
    if target_fname:
        params["target_files"] = build_file_list(target_fname, input_spec)
    params["decoder_mask_files"] = build_file_list(mask_fname, input_spec)
    
    input_pipeline["params"] = params
    output_spec[pipeline_name] = input_pipeline

def build_file_list(fnames, input_spec):
    if not isinstance(fnames, list):
        # print input_spec["required_files"]
        return input_spec["required_files"][fnames]
    nested = [input_spec["required_files"][fname] for fname in fnames]
    flattened = [item for sublist in nested for item in sublist]
    return flattened


def generate_infer_spec(spec, split="dev"):
    output_spec = {}

    make_input_pipeline("input_pipeline", spec, output_spec,
                        "%s/%s_encode.txt" %(split, split),
                        #"%s/%s_schema_locations.txt" %(split, split),
                        None,
                       None,
                       "%s/%s_decoder_mask.txt" %(split, split))

    if "tasks" in spec:
        output_spec["tasks"] = spec["tasks"]
    else:
        output_spec["tasks"] = [{"class": "DecodeText"}]

    output_spec["model_dir"] = spec["model_dir"]

    # Use defaults from train.py for these, unless we overwrite them
    # in the config file.
    usually_defaults = ["model_params", "checkpoint_path",
                        "batch_size"]
    for param in usually_defaults:
        if param in spec:
            output_spec[param] = spec[param]

    return output_spec

def write_specs(spec):
    model_dir = spec["model_dir"]
    print("Saving yaml specifications to %s."
          % model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    config_dir = os.path.join(model_dir, "config")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    train_spec = generate_training_spec(spec)
    train_file_path = os.path.join(config_dir, "train.yml")
    with open(train_file_path, 'w') as f:
        yaml.dump(train_spec, f, default_flow_style=False)

    train_on_both_spec = generate_training_spec(spec, include_dev=True)
    train_both_file_path = os.path.join(config_dir, "train_with_dev.yml")
    with open(train_both_file_path, 'w') as f:
        yaml.dump(train_on_both_spec, f, default_flow_style=False)

    infer_spec = generate_infer_spec(spec, "dev")
    infer_file_path = os.path.join(config_dir, "infer.yml")
    with open(infer_file_path, 'w') as f:
        yaml.dump(infer_spec, f, default_flow_style=False)

    infer_train_spec = generate_infer_spec(spec, "train")
    infer_train_file_path = os.path.join(config_dir, "infer_train.yml")
    with open(infer_train_file_path, 'w') as f:
        yaml.dump(infer_train_spec, f, default_flow_style=False)

    infer_test_spec = generate_infer_spec(spec, "test")
    infer_test_file_path = os.path.join(config_dir, "infer_test.yml")
    with open(infer_test_file_path, 'w') as f:
        yaml.dump(infer_test_spec, f, default_flow_style=False)

def write_bash_scripts(path, spec):
    model_dir = spec["model_dir"]
    dev_questions = " ".join(spec["required_files"]["dev/dev_encode.txt"])
    test_questions = " ".join(spec["required_files"]["test/test_encode.txt"])
    train_questions = " ".join(spec["required_files"]["train/train_encode.txt"])

    dev_decode = "dev/dev_decode.txt"
    train_decode = "train/train_decode.txt"
    test_decode = "test/test_decode.txt"
    # If it's a schema copying or both model, have it use the preprocessed
    # input files.
    if spec["model"] in {"SchemaAttentionCopyingSeq2Seq",
                 "SchemaAndInputAttentionCopyingSeq2Seq"}:
        dev_decode = "dev/dev_copy_decode.txt"
        train_decode = "train/train_copy_decode.txt"
        test_decode = "test/test_copy_decode.txt"
    dev_gold = " ".join(spec["required_files"][dev_decode])
    test_gold = " ".join(spec["required_files"][test_decode])
    train_gold = " ".join(spec["required_files"][train_decode])

    script = """
    #!/bin/bash

    cwd=$(pwd)
    cd %s/
    export MODEL_DIR=%s
    export CONFIG_DIR=$MODEL_DIR/config
    DEV_QUESTIONS="%s"
    DEV_GOLD="%s"
    TRAIN_QUESTIONS="%s"
    TRAIN_GOLD="%s"

    echo $MODEL_DIR >> $MODEL_DIR/experiment_log.txt
    echo "Starting train: $(date)" >> $MODEL_DIR/experiment_log.txt
    python -m bin.train --config_paths="$CONFIG_DIR/train.yml"
    echo "$(date) $MODEL_DIR" >> experiments_to_review.txt
    cd $cwd
    """ % (path, model_dir, dev_questions, dev_gold, train_questions, train_gold)

    script_fname = os.path.join(model_dir, "experiment.sh")
    with open(script_fname, 'w') as f:
        f.write(script)
    os.chmod(script_fname, 0775)
    
    
    script_infer = """
    #!/bin/bash

    cwd=$(pwd)
    cd %s/
    export MODEL_DIR=%s
    export CONFIG_DIR=$MODEL_DIR/config
    DEV_QUESTIONS="%s"
    DEV_GOLD="%s"
    TRAIN_QUESTIONS="%s"
    TRAIN_GOLD="%s"

    echo $MODEL_DIR >> $MODEL_DIR/experiment_log.txt
    echo "Starting infer: $(date)" >> $MODEL_DIR/experiment_log.txt
    python -m bin.infer --config_path="$CONFIG_DIR/infer.yml" > $MODEL_DIR/output.txt
    echo "Starting infer on test: $(date)" >> $MODEL_DIR/experiment_log.txt
    python -m bin.infer --config_path="$CONFIG_DIR/infer_test.yml" > $MODEL_DIR/output_test.txt
    echo "Starting infer on train: $(date)" >> $MODEL_DIR/experiment_log.txt
    python -m bin.infer --config_path="$CONFIG_DIR/infer_train.yml" > $MODEL_DIR/output_train.txt
    echo "Starting evaluation: $(date)" >> $MODEL_DIR/experiment_log.txt
    echo "$(date) $MODEL_DIR" >> experiments_to_review.txt
    cd $cwd
    """ % (path, model_dir, dev_questions, dev_gold, train_questions, train_gold)

    script_fname = os.path.join(model_dir, "experiment_infer.sh")
    with open(script_fname, 'w') as f:
        f.write(script_infer)
    os.chmod(script_fname, 0775)
    
    

    test_script = """
    #!/bin/bash

    cwd=$(pwd)
    cd %s/
    export MODEL_DIR=%s
    export CONFIG_DIR=$MODEL_DIR/config
    TEST_QUESTIONS="%s"
    TEST_GOLD="%s"

    echo $MODEL_DIR >> $MODEL_DIR/experiment_log.txt
    echo "Starting infer on test: $(date)" >> $MODEL_DIR/experiment_log.txt
    python -m bin.infer --config_path="$CONFIG_DIR/infer_test.yml" > $MODEL_DIR/test_output.txt
    echo "Starting evaluation of test: $(date)" >> $MODEL_DIR/experiment_log.txt
    python quick_eval.py -q $TEST_QUESTIONS -g $TEST_GOLD -s $MODEL_DIR/test_output.txt > $MODEL_DIR/quick_eval_test.txt
    echo "$(date) TEST $MODEL_DIR" >> experiments_to_review.txt
    cd $cwd
    """ % (path, model_dir, test_questions, test_gold)

    script_fname = os.path.join(model_dir, "run_test.sh")
    with open(script_fname, 'w') as f:
        f.write(test_script)
    os.chmod(script_fname, 0775)

    curves_script = """
    #!/bin/bash
    export MODEL_DIR=%s
    export CONFIG_DIR=$MODEL_DIR/config
    export PRED_DIR=$MODEL_DIR/predictions
    mkdir -p $PRED_DIR
    DEV_GOLD="%s"
    TRAIN_GOLD="%s"

    cwd=$(pwd)
    cd %s/
    for fname in $MODEL_DIR/model.ckpt-*.index
    do
        echo $fname
        tmp=${fname#*ckpt-}
        num=${%s}
        if [ -s ${PRED_DIR}/predictions-$num.txt ]
        then
            echo "Already evaluated."
        else
            python -m bin.infer \\
              --config_path="$CONFIG_DIR/infer.yml"\\
              --checkpoint_path ${MODEL_DIR}/model.ckpt-$num \\
              >  ${PRED_DIR}/predictions-$num.txt

            echo "==================TRAIN================" \\
              >> ${PRED_DIR}/predictions-$num.txt

            python -m bin.infer \\
              --config_path="$CONFIG_DIR/infer_train.yml"\\
              --checkpoint_path ${MODEL_DIR}/model.ckpt-$num \\
              >>  ${PRED_DIR}/predictions-$num.txt
        fi
    done
    python plot_training_curves.py $PRED_DIR -t $TRAIN_GOLD -d $DEV_GOLD
    cd $cwd
    """  %(model_dir, dev_gold, train_gold, path, "tmp%.*")
    script_fname = os.path.join(model_dir, "plot_training_curves.sh")
    with open(script_fname, 'w') as f:
        f.write(curves_script)
    os.chmod(script_fname, 0775)

def main():
    parser = argparse.ArgumentParser(description="Build config files "
                                     " for the specified experiments.")
    parser.add_argument("spec_file", help="Specification file path")
    args = parser.parse_args()
    simple_spec = parse_yaml(args.spec_file)
    expanded_spec = expand_spec(simple_spec)
    write_specs(expanded_spec)
    path = os.getcwd()
    write_bash_scripts(path, expanded_spec)

if __name__=="__main__":
    main()
