/**
 * Created by kai on 2018/4/18.
 */

var split = process.argv[2]
console.log(split)

if (split == "data_radn_train") {
  var input_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/data_radn_split/train_gold.sql";
  var output_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/data_radn_split/train-ast.json";
} else if (split == "data_radn_dev") {
  var input_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/data_radn_split/dev_gold.sql";
  var output_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/data_radn_split/dev-ast.json";
} else if (split == "data_radn_test") {
  var input_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/data_radn_split/test_gold.sql";
  var output_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/data_radn_split/test-ast.json";
} else if (split == "data_train") {
  var input_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/data/train_gold.sql";
  var output_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/data/train-ast.json";
} else if (split == "data_dev") {
  var input_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/data/dev_gold.sql";
  var output_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/data/dev-ast.json";
} else if (split == "data_test") {
  var input_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/data/test_gold.sql";
  var output_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/data/test-ast.json";
} else if (split == "bad") {
  var input_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/bad.sql"
  var output_file_path = "/home/lily/rz268/seq2sql/baselines_rui/nl2code/data_preprocess/bad-ast.json"

}
console.log(input_file_path)
console.log(output_file_path)

var fs = require('fs');
var sqliteParser = require('sqlite-parser');
var parserTransform = sqliteParser.createParser();
var singleNodeTransform = sqliteParser.createStitcher();
var readStream = fs.createReadStream(input_file_path);
var writeStream = fs.createWriteStream(output_file_path);
var liner = require('./liner');
readStream.pipe(liner);
liner.pipe(parserTransform);
//readStream.pipe(parserTransform);
//parserTransform.pipe(process.stdout);
parserTransform.pipe(singleNodeTransform);
singleNodeTransform.pipe(writeStream);

var len = 0;
var readline = 0;
parserTransform.on('error', function (err) {
  var data = this.read();
  if(data){
    //console.log(data.toString());
    //data = JSON.parse(data.toString()) 
    //len += len(data)
	
	  len += data.toString().split("}{").length
  }
  var startLine =len;
  var location = "[" + startLine +
        ", " + err.location.start.column + "] ";
  console.log(location + err.message);
  console.error(err);
  process.exit(1);
});
/*
readStream.on("data",function(data){
  console.log(data.toString('utf8'));
  var size = data.toString('utf8').split(/\r\n|\r|\n/).length;
  readline += size;
  console.log(readline);
  process.exit(1);
});*/
//readStream.on('readable',function(){
//  console.log(this.read().toString('utf8'));
//})
//
//liner.on("data",function(data){
//  readline++;
//})
parserTransform.on('data', function(data){
 // var size = data.toString('utf8').split(/\r\n|\r|\n/).length;
 // len += size;
 // console.log(data.toString('utf8'));
  len++;
  console.log(len);
})

writeStream.on('finish', function () {
  process.exit(0);
});
