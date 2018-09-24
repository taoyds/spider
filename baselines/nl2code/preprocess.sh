#! /bin/bash

python2 lang/sql/sql_dataset.py \
 -table_schema ./data_preprocess/data_radn_split/tables.json \
 -train_data ./data_preprocess/data_radn_split/train_radn.json \
 -train_data_ast ./data_preprocess/data_radn_split/train-ast.json \
 -dev_data ./data_preprocess/data_radn_split/dev_radn.json \
 -dev_data_ast ./data_preprocess/data_radn_split/dev-ast.json \
 -test_data ./data_preprocess/data_radn_split/test_radn.json \
 -test_data_ast ./data_preprocess/data_radn_split/test-ast.json \
 -train_data_size 8659 \
 -dev_data_size 1034 \
 -output_path ./data_preprocess/dataset_radn.bin

python2 lang/sql/sql_dataset.py \
 -table_schema ./data_preprocess/data/tables.json \
 -train_data ./data_preprocess/data/train.json \
 -train_data_ast ./data_preprocess/data/train-ast.json \
 -dev_data ./data_preprocess/data/dev.json \
 -dev_data_ast ./data_preprocess/data/dev-ast.json \
 -test_data ./data_preprocess/data/test.json \
 -test_data_ast ./data_preprocess/data/test-ast.json \
 -train_data_size 8659 \
 -dev_data_size 1034 \
 -output_path ./data_preprocess/dataset.bin
