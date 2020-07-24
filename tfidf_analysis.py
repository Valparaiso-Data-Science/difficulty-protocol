#IN:
 #dictionary: dictionary being accessed
 #categories: overall categories being processed (questions or roles)
 #subcategories: subcategories being processed (roles or questions -- opposite of categories)
#No OUT, creates csv file for each category in categories

import pandas as pd
import spacy
import string
import re
from re import *
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
def tfidf(dictionary, categories, subcategories):
  topics = list(categories) #questions or roles
  nlp = spacy.load("en")
  answers = dictionary
  for topic in topics:
    print("Grouping: %s -----------------" % topic)
    sub = subcategories
    corpus = []
    #initalize documents in corpus
    for i in range(len(sub)):
      category = sub[i]
      corpus.append(answers[topic][category])
      corpus[i] = str(corpus[i])[1:-1].replace("'","")

    #code from https://www.geeksforgeeks.org/tf-idf-for-bigrams-trigrams/
    #GETTING BIGRAMS
    vectorizer = CountVectorizer(ngram_range = (2,2))
    X1 = vectorizer.fit_transform(corpus)
    features = (vectorizer.get_feature_names())
    # Applying TFIDF
    vectorizer = TfidfVectorizer(ngram_range = (2,2))
    X2 = vectorizer.fit_transform(corpus)
    scores = (X2.toarray())
    # Getting top ranking features
    sums = X2.sum(axis = 0)
    data1 = []
    for col, term in enumerate(features):
        data1.append( (term, sums[0,col] ))
    ranking = pd.DataFrame(data1, columns = ['term','rank'])
    words = (ranking.sort_values('rank', ascending = False))
    words = words[:10]

    #Getting frequency counts
    freq_list = []
    list_bigram = list(words['term'])
    i = 0
    while i < 10:
      #getting words in bigram separately
      bigram = str(list_bigram[i]).split()
      f_word = str(bigram[0])
      s_word = str(bigram[1])
      regex = re.escape(f_word) + r" " + re.escape(s_word) #regex of bigram words with space in between
      count = 0
      for doc in corpus:
        iterator = finditer(regex, doc) #search documents for matching regex
        for match in iterator:
          count+=1
      app = str(count)
      freq_list.append(app)
      i+=1
    words['frequency'] = freq_list
    denom = 0
    for category in sub:
      denom += len(answers[topic][category])
    words ['overall'] = str(denom)
    words['category'] = topic
    words = words[['category','term','frequency','overall','rank']]
    print('\n',words)
    print("\n---------------------------\n")

    #saving dataframe to csv
    path_name = topic+'.csv'
    words.to_csv(data_path+path_name)
