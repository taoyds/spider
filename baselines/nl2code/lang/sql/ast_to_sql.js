/**
 * Created by kai on 2018/4/21.
 */
import { generate } from './sqlgenerate-babel';
var fs = require('fs');

var split = process.argv[2]
console.log(split)

if (split == "data_radn") {
    var input_path = "./decode_data_radn_result.json"
    var output_path = "./data_radn_result.txt"
} else if (split == "data") {
    var input_path = "./decode_data_result.json"
    var output_path = "./data_result.txt"
}

var obj = JSON.parse(fs.readFileSync(input_path, 'utf8'));
var writer = fs.createWriteStream(output_path);
console.log(obj.length);
for(var i = 0;i<obj.length;i++){
    var sql = "";
    try {
        // console.log(obj[i][0][1]);
        sql = generate(obj[i][0][1]);
        // console.log(sql.replace("\n"," "));
        // sql = sql.replace("\n"," ")
        var re=/\r\n|\n\r|\n|\r/g;
        sql = sql.replace(re," ")
    } catch (e){
        sql = "select a from b";
    } finally {
        writer.write(sql+"\n");
    }
}
