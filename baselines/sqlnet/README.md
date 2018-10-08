
## Modified SQLNet Baseline for the Spider Text-to-SQL Task

Source code of modified SQLNet model reported on our EMNLP 2018 paper:[Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task ](https://arxiv.org/abs/1809.08887) to be updated.


#### Environment Setup

1. The code uses Python 2.7 and [Pytorch 0.2.0](https://pytorch.org/previous-versions/) GPU.
2. Install Python dependency: `pip install -r requirements.txt`


#### Download Data and Embeddings
1. download the dataset from [the Spider task website](https://yale-lily.github.io/spider) to be updated, and put `tables.json`, `train.json`, and `dev.json` under `data/` directory.
2. Download the pretrained [Glove](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip)


#### Train Models

```
  mkdir saved_models
  python train.py --dataset data/
```

#### Test Models

We are not going to release our test dataset. Thus, we run the test script using the development data.
```
python test.py --dataset data/ --output predicted_sql.txt
```

#### Evaluation

Follow the general evaluation process in this github.
