#! /bin/bash

DATA_DIR=/home/lily/rz268/seq2sql/baselines_rui/datasets_final
cd data_preprocess
python2 clean_sql.py $DATA_DIR
cd ..
cp $DATA_DIR/data/*.json ./data_preprocess/data/
cp $DATA_DIR/data_radn_split/*.json ./data_preprocess/data_radn_split/
