#Lemmatizes, removes stop words, creates word tokens
#IN:
 #col: list of sentences from the column
 #stop_words: list of stop words
#OUT:
 #lol: list of lists of tokenized sentences with stop words removed and words lemmatized

import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
 def words(col, stop_words,sub_words):
  nlp = spacy.load('en', disable=['parser', 'ner'])
  lol = []
  for sent in col: #for each sentence
    sent = ' '.join(sent)
    wr_string = sent.lower() #lowercase
    for key, value  in sub_words.items(): #for key value pairings in dictionary
        wr_string = wr_string.replace(str(key), str(value)) #replace the key with the value
    doc = nlp(wr_string)
    new_sent = " ".join([token.lemma_ for token in doc]) #lemmatize words
    new_sent = new_sent.split()
    new_sent = [word for word in new_sent if word not in stop_words] #remove stop words
    lol.append(new_sent)
  return lol
