#! /bin/bash

output="runs_data"
device="cuda0"
dataset="data_preprocess/dataset.bin"
glove="../glove/glove.42B.300d.txt"

commandline="-batch_size 8 -max_epoch 50 -save_per_batch 4000 -decode_max_time_step 100 -optimizer adam -rule_embed_dim 128 -node_embed_dim 64 "
datatype="sql"

 #train the model
THEANO_FLAGS="mode=FAST_RUN,device=${device},floatX=float32,dnn.base_path=/data/lily/rz268/system/cuda-8.0" python2 -u code_gen.py \
	-data_type ${datatype} \
	-data ${dataset} \
  -glove_path ${glove}\
	-output_dir ${output} \
	${commandline} \
	train
