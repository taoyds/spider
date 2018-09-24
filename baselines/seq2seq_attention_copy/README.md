## Seq2Seq Baselines

This is a documentation for for the sequence-to-sequence model on 2 datasets (data, data_radn_split) with 3 methods (basic, attention, attention+copy) each. With mask and tune with glove embedding.

## Dependency

This code is based on Google seq2seq v0.1 and the code by Catherine Finegan-Dollak.

- Python 2.7.14 :: Anaconda custom (64-bit)
- tensorflow, gpu: 1.5.0
- numpy: 1.13.3 
- matplotlib: 2.1.0

## Folder Structure

- `bin/` contains entry point to the model including train.py and infer.py, and other tool codes
- `experimental_configs/` contains experimental configurations in yaml format 
- `data/` contains folder `pre_process`, folder `glove`, folder `datasets` folder containing 2 datasets.
- `seq2seq/` contains the main model code
- `config_builder.py` helps make new model directory and write configurations into bash files

## 1. Configuration

#### 1.1 Prepare Data

Download `glove.6B.100d.txt` in `data/glove/`.

Put the original data in the folder 

```
data/datasets/data             
data/datasets/data_radn_split
```

Then we should generate data to folder 

```
data/datasets/data_processed
data/datasets/data_radn_split_processed
```

by changing the `infiles` and `prefix` variables in `data/pre_process/utils.py`, and use `data/pre_process/generate_vocab.py` to generate processed data.


#### 1.2  Configuration yaml

We run 6 experiments. The configuration yaml files are in `experimental_configs` folder:

- `attn_copying_tune_data_radn_split.yaml` # attention + copy, data_radn_split
- `attn_tune_data.yaml` # attention, data
- `attn_copying_tune_data.yaml` # attention + copy, data
- `basic_tune_data_radn_split.yaml` # basic, data_radn_split
- `attn_tune_data_radn_split.yaml` # attention, data_radn_split
- `basic_tune_data.yaml` # basic, data

In the configuration yaml files, change the data directotries:

- `data_directories`: `data/datasets/data_processed/`
- `embedding.file`: `data/glove/glove.6B.100d.txt`


#### 1.3 Build model folder

Use `config_builder.py` to generate model folder with configuration and bash files using configs in `experimental_configs/`:

```
python config_builder.py [configuration_yaml_file] 
```

Then we will get 6 model folders

- `InputAttentionCopyingSeq2Seq_tune_model_data/`
- `InputAttentionCopyingSeq2Seq_tune_model_data_radn_split/`
- `BasicSeq2Seq_tune_model_data/`
- `BasicSeq2Seq_tune_model_data_radn_split/`
- `AttentionSeq2Seq_tune_model_data/`
- `AttentionSeq2Seq_tune_model_data_radn_split/`


## 2. Run Experiments

For training:

```
./[model_folder]/experiment.sh
```

For testing:

```
./[model_folder]/experiment_infer.sh
```
with the following outputs:

- train: `[model_folder]/output_train.txt`
- dev: `[model_folder]/output.txt`
- test: `[model_folder]/output_test.txt`

## 3. Evaluation

Note before evaluation, need to replace all ' . ' in the output, then compare the result with the 
datasets/ [processed dataset name] / [data split] / [data split] _decode.txt

## Contact

Dongxu Wang, Rui Zhang
