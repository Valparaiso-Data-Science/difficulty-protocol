#IN:
 #csv: csv name
 #og_name: original column name
 #new_name: new column name
import pandas as pd
def rename_col(csv, og_name, new_name):
    csv.rename(columns={og_name:new_name},inplace=True)
