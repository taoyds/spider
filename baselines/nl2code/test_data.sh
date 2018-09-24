#! /bin/bash

output="runs_data"
device="cuda0"
dataset="data_preprocess/dataset.bin"
glove="../glove/glove.42B.300d.txt"

commandline="-batch_size 10 -max_epoch 50 -save_per_batch 4000 -decode_max_time_step 100 -optimizer adam -rule_embed_dim 128 -node_embed_dim 64 "
datatype="sql"

model="model-epoch43.npz"
# decode testing set, and evaluate the model which achieves the best bleu and accuracy, resp.
#for model in "model.best_bleu.npz" "model.best_acc.npz"; do
THEANO_FLAGS="mode=FAST_RUN,device=${device},floatX=float32,dnn.base_path=/data/lily/rz268/system/cuda-8.0" python code_gen.py \
	-data_type ${datatype} \
	-data ${dataset} \
  -glove_path ${glove}\
	-output_dir ${output} \
	-model ${output}/${model} \
  -ast_output_path ./decode_data_result.json \
	${commandline} \
	decode \
	-saveto ${output}/${model}.decode_results.test.bin \
