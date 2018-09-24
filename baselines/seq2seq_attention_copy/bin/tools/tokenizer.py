from nltk.tokenize import word_tokenize
import collections

train_file_en = "train_encode.txt"
train_file_de = "train_decode.txt"
infile = "/home/lily/dw633/seq2seq/seq2sql/seq2seq/data/data/train/"

minimum = 3

cnt_en = collections.Counter()
cnt_de = collections.Counter()
with open(infile+train_file_en) as inf: 
    for line in inf:
        words = word_tokenize(line)
        # print words
        for w in words:
            cnt_en[w] += 1

print (cnt_en)

with open(infile+train_file_de) as inf: 
    for line in inf:
        words = word_tokenize(line)
        # print words
        for w in words:
            cnt_de[w] += 1

print (cnt_de)

with open(infile+"encode_vocab.txt", "w") as outf:
    for key in cnt_en:
        print (key)
        outf.write(key + "\n")
        
        
with open(infile+"decode_vocab.txt", "w") as outf:
    for key in cnt_de:
        outf.write(key + "\n")