#! /bin/bash

node lang/sql/parser-ast.js data_radn_train
node lang/sql/parser-ast.js data_radn_dev
node lang/sql/parser-ast.js data_radn_test
#
node lang/sql/parser-ast.js data_train
node lang/sql/parser-ast.js data_dev
node lang/sql/parser-ast.js data_test
