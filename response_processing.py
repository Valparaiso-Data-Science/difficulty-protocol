#IN:
 #stops: list of stop words
 #qs: list of questions
 #roles: list of dataframes
#No OUT, creates two dictionaries containing responses ordered for usage in TF IDF analysis

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from dataframe_name import get_df_name
def process_responses(stops,sub,qs,dfs):
  answers = {} #first dictionary organization
  for q in qs:
    answers[q] = {}
  answers2 = {} #second dictionary organization
  for df in dfs:
    answers2[get_df_name(df)] = {}
  for df in dfs: #loop through dataframes
    for q in qs: #loop through questions
      for col in df: #loop through columns in current dataframe
        if col == q: #if column name mataches current question
          sentences = tokenize(df[col])
          temp_list = words(sentences,stops,sub)
          i=0
          for lst in temp_list:
            joined = ' '.join(lst)
            temp_list[i] = joined
            i+=1
          temp_list = list(filter(None, temp_list))
          temp_list[:] = [item for item in temp_list if item != 'nan']
          answers[q][get_df_name(df)] = temp_list
          answers2[get_df_name(df)][col] = temp_list
  return answers, answers2
