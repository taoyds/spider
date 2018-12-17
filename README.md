# Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task

Spider is a large human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task (natural language interfaces for relational databases). It is released along with our EMNLP 2018 paper: [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887). This repo contains all code for evaluation, preprocessing, and all baselines used in our paper. Please refer to [the task site](https://yale-lily.github.io/spider) for more general introduction and the leaderboard.


### Changelog

- `12/17/2018` We updated 7 sqlite database files. Please download the Spider data from the official website again. Please refer to [the issue 14](https://github.com/taoyds/spider/issues/14) for more details.
- `10/25/2018`: evaluation script is updated so that the table in `count(*)`cases will be evaluated as well. Please check out [the issue 5](https://github.com/taoyds/spider/issues/5) for more info. Results of all baselines and [syntaxSQL](https://github.com/taoyds/syntaxSQL) on the papers are updated as well.
- `10/25/2018`: to get the latest SQL parsing results (a few small bugs fixed), please use `preprocess/parse_raw_json.py` to update. Please refer to [the issue 3](https://github.com/taoyds/spider/issues/3) for more details.

### Citation

The dataset is annotated by 11 college students. When you use the Spider dataset, we would appreciate it if you cite the following:

```
@inproceedings{Yu&al.18c,
  year =         2018,
  title =        {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  booktitle =    {EMNLP},
  author =       {Tao Yu and Rui Zhang and Kai Yang and Michihiro Yasunaga and Dongxu Wang and Zifan Li and James Ma and Irene Li and Qingning Yao and Shanelle Roman and Zilin Zhang and Dragomir Radev }
}
```

### Installation

`evaluation.py` and `process_sql.py` are written in Python 2.7. Enviroment setup for each baseline is in README under each baseline directory.


### Data Content and Format

#### Question, SQL, and Parsed SQL

Each file in`train.json` and `dev.json` contains the following fields:
- `question`: the natural language question
- `question_toks`: the natural language question tokens
- `db_id`: the database id to which this question is addressed.
- `query`: the SQL query corresponding to the question. 
- `query_toks`: the SQL query tokens corresponding to the question. 
- `sql`: parsed results of this SQL query using `process_sql.py`. Please refer to `parsed_sql_examples.sql` in the`preprocess` directory for the detailed documentation.


```
 {
        "db_id": "world_1",
        "query": "SELECT avg(LifeExpectancy) FROM country WHERE Name NOT IN (SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  \"English\" AND T2.IsOfficial  =  \"T\")",
        "query_toks": ["SELECT", "avg", "(", "LifeExpectancy", ")", "FROM", ...],
        "question": "What is average life expectancy in the countries where English is not the official language?",
        "question_toks": ["What", "is", "average", "life", ...],
        "sql": {
            "except": null,
            "from": {
                "conds": [],
                "table_units": [
                    ...
            },
            "groupBy": [],
            "having": [],
            "intersect": null,
            "limit": null,
            "orderBy": [],
            "select": [
                ...
            ],
            "union": null,
            "where": [
                [
                    true,
                    ...
                    {
                        "except": null,
                        "from": {
                            "conds": [
                                [
                                    false,
                                    2,
                                    [
                                    ...
                        },
                        "groupBy": [],
                        "having": [],
                        "intersect": null,
                        "limit": null,
                        "orderBy": [],
                        "select": [
                            false,
                            ...
                        "union": null,
                        "where": [
                            [
                                false,
                                2,
                                [
                                    0,
                                   ...
        }
    },

```

#### Tables

`tables.json` contains the following information for each database:
- `db_id`: database id
- `table_names_original`: original table names stored in the database.
- `table_names`: cleaned and normalized table names. We make sure the table names are meaningful. [to be changed]
- `column_names_original`: original column names stored in the database. Each column looks like: `[0, "id"]`. `0` is the index of table names in `table_names`, which is `city` in this case. `"id"` is the column name. 
- `column_names`: cleaned and normalized column names. We make sure the column names are meaningful. [to be changed]
- `column_types`: data type of each column
- `foreign_keys`: foreign keys in the database. `[3, 8]` means column indices in the `column_names`. These two columns are foreign keys of two different tables.
- `primary_keys`: primary keys in the database. Each number is the index of `column_names`.


```
{
    "column_names": [
      [
        0,
        "id"
      ],
      [
        0,
        "name"
      ],
      [
        0,
        "country code"
      ],
      [
        0,
        "district"
      ],
      .
      .
      .
    ],
    "column_names_original": [
      [
        0,
        "ID"
      ],
      [
        0,
        "Name"
      ],
      [
        0,
        "CountryCode"
      ],
      [
        0,
        "District"
      ],
      .
      .
      .
    ],
    "column_types": [
      "number",
      "text",
      "text",
      "text",
         .
         .
         .
    ],
    "db_id": "world_1",
    "foreign_keys": [
      [
        3,
        8
      ],
      [
        23,
        8
      ]
    ],
    "primary_keys": [
      1,
      8,
      23
    ],
    "table_names": [
      "city",
      "sqlite sequence",
      "country",
      "country language"
    ],
    "table_names_original": [
      "city",
      "sqlite_sequence",
      "country",
      "countrylanguage"
    ]
  }
```


#### Databases

All table contents are contained in corresponding SQLite3 database files.


### Evaluation

Our evaluation metrics include Component Matching, Exact Matching, and Execution Accuracy. For component and exact matching evaluation, instead of simply conducting string comparison between the predicted and gold SQL queries, we decompose each SQL into several clauses, and conduct set comparison in each SQL clause. 

For Execution Accuracy, our current models do not predict any value in SQL conditions so that we do not provide execution accuracies. However, we encourage you to provide it in the future submissions. For value prediction, you can assume that a list of gold values for each question is given. Your model has to fill them into the right slots in the SQL.

Please refer to [our paper]() and [this page](https://github.com/taoyds/spider/tree/master/evaluation) for more details and examples.

```
python evaluation.py --gold [gold file] --pred [predicted file] --etype [evaluation type] --db [database dir] --table [table file]

arguments:
  [gold file]        gold.sql file where each line is `a gold SQL \t db_id`
  [predicted file]   predicted sql file where each line is a predicted SQL
  [evaluation type]  "match" for exact set matching score, "exec" for execution score, and "all" for both
  [database dir]     directory which contains sub-directories where each SQLite3 database is stored
  [table file]       table.json file which includes foreign key info of each database
  
```

### FAQ



