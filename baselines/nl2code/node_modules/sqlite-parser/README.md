# sqlite-parser

[![NPM Version Image](https://img.shields.io/npm/v/sqlite-parser.svg)](https://www.npmjs.com/package/sqlite-parser)
[![dependencies Status Image](https://img.shields.io/david/codeschool/sqlite-parser.svg)](https://github.com/codeschool/sqlite-parser/)
[![devDependencies Status Image](https://img.shields.io/david/dev/codeschool/sqlite-parser.svg)](https://github.com/codeschool/sqlite-parser/)
[![License Type Image](https://img.shields.io/github/license/codeschool/sqlite-parser.svg)](https://github.com/codeschool/sqlite-parser/blob/master/LICENSE)

This JavaScript library parses SQLite queries to generate
_abstract syntax tree_ (AST) representations of the parsed statements.

Try out the
[interactive demo](http://codeschool.github.io/sqlite-parser/demo/)
to see it in action.

**This parser is written against the [SQLite 3 spec](https://www.sqlite.org/lang.html).**

## Install

```
npm install sqlite-parser
```

### Install as a global module **(since v1.0.0)**

Use the command-line interface of the parser by installing it as a global module.
The `sqlite-parser` command is then available to use to parse input SQL files and
write the results to stdout or a JSON file. Additional usage
instructions and options available through `sqlite-parser --help`.

```
npm i -g sqlite-parser
```

## Basic Usage

The library exposes a function that accepts two arguments: a string containing
SQL to parse and a callback function. If an AST cannot be generated from the
input string then a descriptive error is generated.

If invoked without a callback function the parser will runs synchronously and
return the resulting AST or throw an error if one occurs.

``` javascript
var sqliteParser = require('sqlite-parser');
var query = 'select pants from laundry;';
// sync
var ast = sqliteParser(query);
console.log(ast);

// async
sqliteParser(query, function (err, ast) {
  if (err) {
    console.error(err);
    return;
  }
  console.log(ast);
});
```

## Use parser on Node streams *(experimental)* **(since v1.0.0)**

This library also includes *experimental* support as a
[stream transform](https://nodejs.org/api/stream.html) that can accept a
_readable_ stream of SQL statements and produce a JSON string, representing
the AST of each statement, as it is read and transformed. Using this method,
the parser can handle files containing hundreds or thousands of queries at
once without running into memory limitations. The AST for each statement is
pushed down the stream as soon as it is read and parsed instead of reading the
entire file into memory before parsing begins.

``` javascript
var parserTransform = require('sqlite-parser').createParser();
var readStream = require('fs').createReadStream('./large-input-file.sql');

readStream.pipe(parserTransform);
parserTransform.pipe(process.stdout);

parserTransform.on('error', function (err) {
  console.error(err);
  process.exit(1);
});

parserTransform.on('finish', function () {
  process.exit(0);
});
```

To pipe the output into a file that contains a single valid JSON structure, the
output of the parser steam transform needs to be wrapped in statement list node
where every statement is separated by a comma.

``` javascript
var fs = require('fs');
var sqliteParser = require('sqlite-parser');
var parserTransform = sqliteParser.createParser();
var singleNodeTransform = sqliteParser.createStitcher();
var readStream = fs.createReadStream('./large-input-file.sql');
var writeStream = fs.createWriteStream('./large-output-file.json');

readStream.pipe(parserTransform);
parserTransform.pipe(singleNodeTransform);
singleNodeTransform.pipe(writeStream);

parserTransform.on('error', function (err) {
  console.error(err);
  process.exit(1);
});

writeStream.on('finish', function () {
  process.exit(0);
});
```

## AST

The AST is stable as of release `1.0.0`. However, if changes need to be made to
improve consistency between node types, they will be explicitly listed in the
[CHANGELOG](https://github.com/codeschool/sqlite-parser/blob/master/CHANGELOG.md).

### Example

You can provide one or more SQL statements at a time. The resulting AST object
has, at the highest level, a statement list node that contains an array of
statements.

#### Input SQL

``` sql
SELECT
 MAX(honey) AS "Max Honey"
FROM
 BeeHive
```

#### Result AST

``` json
{
  "type": "statement",
  "variant": "list",
  "statement": [
    {
      "type": "statement",
      "variant": "select",
      "result": [
        {
          "type": "function",
          "name": {
            "type": "identifier",
            "variant": "function",
            "name": "max"
          },
          "args": {
            "type": "expression",
            "variant": "list",
            "expression": [
              {
                "type": "identifier",
                "variant": "column",
                "name": "honey"
              }
            ]
          },
          "alias": "Max Honey"
        }
      ],
      "from": {
        "type": "identifier",
        "variant": "table",
        "name": "beehive"
      }
    }
  ]
}
```

## Syntax Errors

This parser will try to create *descriptive* error messages when it cannot parse
some input SQL. In addition to an approximate location for the syntax error,
the parser will attempt to describe the area of concern
(e.g.: `Syntax error found near Column Identifier (WHERE Clause)`).

## Contributing

Contributions are welcome! You can get started by checking out the
[contributing guidelines](https://github.com/codeschool/sqlite-parser/blob/master/CONTRIBUTING.md).
