#IN: list of dataframes, list of categories, list of stop words, color for highlighting
#no OUT, prints color coded responses
#all import statements might not be used in this function
import colored
from colored import fg, bg, attr
import pandas as pd
import numpy as np
import nltk
import spacy
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tokenizer import tokenize
from word_processing import words
from dataframe_name import get_df_name

def quality(dfs, categories, stop_words, color):
  topics = list(categories)
  for role in dfs: #for each role
    sub = categories
    good_total = 0
    bad_total = 0
    good = []
    bad = []
    for q in categories: #for each question
      for col in role: #for each response
        if q == col: #if in correct column
          print("%s %s" % (get_df_name(role), q)) #print role and question
          good_words = 0
          bad_words = 0
          tokenized = tokenize(role[col]) #tokenize responses
          processed = words(tokenized) #process words
          for lst in processed: #for each sentence
            for word in lst: #for each word
              word = word.lower()
              if word not in stop_words: #if it is not a stop word
                col = fg('black') + bg(good_word_hex_color) #color the word
                res = attr('reset')
                print(col + word + res, end=' ') #print word
                good_words += 1
              else:
                print(word, end=' ') #print word
                bad_words += 1
            print()
          good.append(good_words)
          bad.append(bad_words)
          print("Good words: ", str(good_words)+'/'+str(good_words+bad_words), '--', '%.3f'%(good_words/(good_words+bad_words)*100)+'%') #print percentage of "good" words
          print("Stop words: ", str(bad_words)+'/'+str(good_words+bad_words), '--', '%.3f'%(bad_words/(good_words+bad_words)*100)+'%') #print percentage of "bad" words
          print('\n\n')
    for g in good: #total good
      good_total += g
    #print(get_df_name(role) + " Total Good Words: " + str(good_total))
    for b in bad: #total bad
      bad_total += b
    #print(get_df_name(role) + " Total Stop Words: " + str(bad_total))
    print("%s Total Good Words: " % get_df_name(role), str(good_total)+'/'+str(good_total+bad_total), '--', '%.3f'%(good_total/(good_total+bad_total)*100)+'%')
    print("%s Total Bad Words: " % get_df_name(role), str(bad_total)+'/'+str(bad_total+good_total), '--', '%.3f'%(bad_total/(good_total+bad_total)*100)+'%')
    print('\n \n \n')
