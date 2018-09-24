import os
import sys
import shutil


def clean_sql_file(input_sql_file,output_sql_file):
  print(input_sql_file)
  f = open(input_sql_file)
  f_out = open(output_sql_file,'w')
  cnt = 0
  for line in f.readlines():
    sql,db = line.split('\t')
    if not sql.endswith(';'):
      sql = sql+';'
    f_out.write(sql+'\n')
    cnt += 1
  f.close()
  f_out.close()

  print(output_sql_file)
  print(cnt)
  print()


if __name__=='__main__':
  data_set_dir = sys.argv[1]

  radn_train_sql_file = os.path.join(data_set_dir,'data_radn_split','train_gold.sql')
  radn_dev_sql_file = os.path.join(data_set_dir,'data_radn_split','dev_gold.sql')
  radn_test_sql_file = os.path.join(data_set_dir,'data_radn_split','test_gold.sql') 

  output_data_dir = 'data_radn_split'
  if os.path.isdir(output_data_dir):
    shutil.rmtree(output_data_dir)
  os.mkdir(output_data_dir)
  clean_sql_file(radn_train_sql_file,'{}/train_gold.sql'.format(output_data_dir))
  clean_sql_file(radn_dev_sql_file,'{}/dev_gold.sql'.format(output_data_dir))
  clean_sql_file(radn_test_sql_file,'{}/test_gold.sql'.format(output_data_dir))

  train_sql_file = os.path.join(data_set_dir,'data','train_gold.sql')
  dev_sql_file = os.path.join(data_set_dir,'data','dev_gold.sql')
  test_sql_file = os.path.join(data_set_dir,'data','test_gold.sql')

  output_data_dir = 'data'
  if os.path.isdir(output_data_dir):
    shutil.rmtree(output_data_dir)
  os.mkdir(output_data_dir)
  clean_sql_file(train_sql_file,'{}/train_gold.sql'.format(output_data_dir))
  clean_sql_file(dev_sql_file,'{}/dev_gold.sql'.format(output_data_dir))
  clean_sql_file(test_sql_file,'{}/test_gold.sql'.format(output_data_dir))
