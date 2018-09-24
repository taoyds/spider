# NL2code

A syntactic neural model for parsing natural language to executable code [paper](https://arxiv.org/abs/1704.01696). 

```
@inproceedings{yin17acl,
    title = {A Syntactic Neural Model for General-Purpose Code Generation},
    author = {Pengcheng Yin and Graham Neubig},
    booktitle = {The 55th Annual Meeting of the Association for Computational Linguistics (ACL)},
    address = {Vancouver, Canada},
    month = {July},
    url = {https://arxiv.org/abs/1704.01696},
    year = {2017}
}
```

## Dependency

* Theano
* vprof
* NLTK 3.2.1
* astor 0.6
* node.js

## Folder Structure

* `lang/sql/` contains SQL specific code for the model.
  * `ast_to_sql.js` converts model generated AST structure into SQL statement.
  * `grammar.py` contains SQL grammar related code.(type of terminal etc)
  * `parse.py` takes AST structure and parse to structure can be used in model and take model generated structure and converted back to AST structure.
  * `parser-ast.js` generates AST structure from SQL.
  * `sql_dataset.py` contains data preprocessor related code. Take data and generated model input.
  * `sqlgenerate.js` SQL generate library used by ast_to_sql.js. (Usually does not need to change this file.)
* `nn/` neural network utilities used by the model

## 1. Preprocess
First, install the package needed by generating SQL library. In the root directory, run 
```bash
npm install
npm run sqlgenerate
npm run ast
```

Second, download the GloVe word embedding at `../glove/glove.42B.300d.txt`.

NOTE: You can skip 1.1 because we already have the output in `data_preprocess/`.
### 1.1 Generate SQL AST
* Change the the following variable in `clean_sql.sh`: 
```
data_set_dir = '/absolute/path/to/datasets_final/'
```
Then clean up SQL input data by run the following in the root directory.
```
./clean_sql.sh
```

This will produce two directories in `data_preprocess/`: `data_radn_split/` and `data/`.

* Run `./sql2ast.sh` to generate AST from SQL.

Warning: when run the above script, it may throw exceptions. This means the input SQL file contains incorrect SQL that could not be parsed. It has to be corrected according to the error message.

In input file one line is one SQL statement ended with a semicolon. Output file is a json file contains AST structure for every input SQL.

### 1.2 Generate Preprocessed Data
Change the configuration in `preprocess.sh`, then run:
```bash
./preprocess.sh
```
Once finished, it will produce `dataset_radn.bin` and `dataset.bin` which will be input to the model.

## 2. Train the Model and Generate Test Ouput
### 2.1 Train The Model
Change the configuration in `train_data_radn.sh` and `train_data.sh`.
On top of the script, `output` is the folder to store trained model. `dataset` is the path to preprocessor generated data file.

To train the model run:
For example-split:
```bash
./train_data_radn.sh
```
For database-split:
```bash
./train_data.sh
```

### 2.2 Generate Test Output
Change the configuration in `test_data_radn.sh` and `test_data.sh`.
`model` is the stored model the decoder program expected to use. `-ast_output_path` is the output path for generated AST json file.

To decode test data, run:
```bash
./test_data.sh
```
or
```bash
./test_data_radn.sh
```

This will produce outputs in AST format.

### 2.3 Generate SQL From AST Structure
To convert AST format to SQL, run the following:
```bash
node ./lang/sql/ast_to_sql-babel.js data
```
or
```bash
node ./lang/sql/ast_to_sql-babel.js data_radn
```

## Contact
Kai Yang, Rui Zhang
