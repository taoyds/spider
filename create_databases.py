###############################################
# python create_databases.py
# Using evaluate_exampels/examples/tables.json
# creates databases/*.sqlite
# Author: Prasad
###############################################
import os
import json
import sqlite3

with open('evaluation_examples/examples/tables.json') as f:
    schema = json.load(f)

    databases = {}

    for data in schema:
        db = data["db_id"]

        if db not in databases:
            databases[db] = { "tables": {}, "foreignkeys": [] }

        cindex = 0
        for (tindex, cname) in data["column_names_original"]:
            if tindex == -1:
                continue
            table = data["table_names_original"][tindex]
            if table not in databases[db]["tables"]:
                databases[db]["tables"][table] = { "columns": [], "primarykeys": [] }
            databases[db]["tables"][table]["columns"].append({"name": cname, "type": data["column_types"][cindex]})
            cindex += 1

        if len(data["primary_keys"]):
            for pindex in data["primary_keys"]:
                c = data["column_names_original"][pindex]
                table = data["table_names_original"][c[0]]
                databases[db]["tables"][table]["primarykeys"].append(c[1])

        if len(data["foreign_keys"]):
            for findex in data["foreign_keys"]:
                src_col = data["column_names_original"][findex[0]]
                ref_col = data["column_names_original"][findex[1]]
                databases[db]["foreignkeys"].append({
                    "table": data["table_names_original"][src_col[0]],
                    "column": src_col[1],
                    "ref_table": data["table_names_original"][ref_col[0]],
                    "ref_column": ref_col[1]
                })

    for db in databases:
        os.makedirs("databases/" + db)
        dsn = "databases/" + db + "/" + db + ".sqlite"
        dbconn = sqlite3.connect(dsn)
        dbcur = dbconn.cursor()

        print (dsn)
        for table in databases[db]["tables"]:
            tablesql = "create table " + table + "("
            cdelim = ""
            for col in databases[db]["tables"][table]["columns"]:
                tablesql += cdelim + '"'+ col["name"] +'" '+ col["type"]
                cdelim = ","
            if len(databases[db]["tables"][table]["primarykeys"]):
                tablesql += ",primary key ("+ ','.join(databases[db]["tables"][table]["primarykeys"])  +")"
            tablesql += ");"
            
            print (tablesql)
            try:
                dbcur.execute(tablesql)
            except Exception as e:
                # Review tables.json spec.
                print ("[ERROR]", e)
                                
        for fkeys in databases[db]["foreignkeys"]:
            altersql = ("alter table {} add key pk_{} {} references {} ({});".format(
                fkeys["table"], fkeys["column"], fkeys["column"], fkeys["ref_table"], fkeys["ref_column"]))
            print (altersql)
            try:
                dbcur.execute(altersql)
            except Exception as e:
                # Review tables.json spec.
                print ("[ERROR]", e)

        dbconn.commit()
        dbconn.close()
        print ()

    f.close()
    print ("databases created.\n")

    print ("Try\npython evaluation.py --gold evaluation_examples/gold_example.txt --pred evaluation_examples/pred_example.txt --etype all --table evaluation_examples/examples/tables.json --db databases")
    
