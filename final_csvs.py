#IN:
 #new_csv: name of new csv file
 #old_csvs: list of old csvs
#No OUT, creates one csv file of merged data from old_csvs

import pandas as pd
def combine_csv(new_csv,old_csvs):
  all_files = old_csvs
  df_merged = pd.concat(all_files, ignore_index=True)
  del(df_merged['Unnamed: 0'])
  del(df_merged['rank'])
  print("Creating 'csvs/"+new_csv".csv'...")
  df_merged.to_csv('csvs/'+new_csv+'.csv',encoding="utf-8-sig",index=False)
