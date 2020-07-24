import pandas as pd
import numpy as np
import nltk
import spacy
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import string
import re
from re import *
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from final_csvs import combine_csv
from rename_columns import rename_col
from response_processing import process_responses
from tfidf_analysis import tfidf
from tokenizer import tokenize
from word_processing import words
from dataframe_name import get_df_name

# load faculty and TA dataset
profta = pd.read_csv("Faculty_TA Difficulty Protocol (Raw).csv", index_col = 0)
stu = pd.read_csv('Student Difficulty Protocol (Raw).csv', index_col = 0)

#Rename Student Columns for ease of reference later
rename_col(stu,'Timestamp','time')
rename_col(stu,'Participant ID','id')
rename_col(stu, 'What topics did you cover in class this week?','topics')
rename_col(stu,'What kinds of activities did you focus on out of class this week?  [Reading/Research]','reading_research')
rename_col(stu, 'What kinds of activities did you focus on out of class this week?  [Continuing/Finishing In-Class Work]','finishing_class_work')
rename_col(stu, 'What kinds of activities did you focus on out of class this week?  [Homework Problems/Assignments]','homework')
rename_col(stu, 'What kinds of activities did you focus on out of class this week?  [Out-of-Class Labs]', 'out_of_class_labs')
rename_col(stu, 'What kinds of activities did you focus on out of class this week?  [Multi-Week Project Experience]','multi_week_project')
rename_col(stu, 'What kinds of activities did you focus on out of class this week?  [Other]','other_outside_class')
rename_col(stu, 'What concepts/activities did you or your peers struggle most with this week?','struggles')
rename_col(stu, 'What questions did you or your peers raise to your instructor/TAs this week?','questions')
rename_col(stu, 'Were there any questions from your peers that surprised you this week?','surprises')

#Rename Prof/TA Columns for ease of reference later
rename_col(profta, 'Timestamp','time')
rename_col(profta, 'What is your role in the course you are reflecting on? ','role')
rename_col(profta, 'What topics did you cover this week?','topics')
rename_col(profta, 'What kinds of activities did your students focus  on this week out of class?  [Reading/Research]','reading_research')
rename_col(profta, 'What kinds of activities did your students focus  on this week out of class?  [Continuing/Finishing In-Class Work]','finishing_class_work')
rename_col(profta, 'What kinds of activities did your students focus  on this week out of class?  [Homework Problems/Assignments]','homework')
rename_col(profta, 'What kinds of activities did your students focus  on this week out of class?  [Out-of-Class Labs]', 'out_of_class_labs')
rename_col(profta, 'What kinds of activities did your students focus  on this week out of class?  [Multi-Week Project Experience]','multi_week_project')
rename_col(profta, 'What kinds of activities did your students focus  on this week out of class?  [Other]','other_outside_class')
rename_col(profta, 'By observation, what concepts or processes did students struggle with?','struggles')
rename_col(profta, 'What questions did students raise this week?','questions')
rename_col(profta, 'Which student questions were surprising to you?','surprises')

# split the profta dataframe into professors and TAs
prof = pd.DataFrame()  # professors
ta = pd.DataFrame()   # TAs

# determine the row indices corresponding to TA answers
ta_inds = []
for i in range(profta.shape[0]):
  ta_inds.append("Teaching Assistant" in profta["role"][i])
ta_inds = np.asarray(ta_inds)

for j in range(profta.values.shape[1]):
  key = profta.keys()[j]

#create two dataframes from profta dataframe -- one for profs, one for tas
  ta[key] = profta.values[:, j][ta_inds]
  prof[key] = profta.values[:, j][np.logical_not(ta_inds)]

#stop words definition
stop_words = list(stopwords.words('english'))
stop_words.extend(list(set(string.punctuation)))
extra_verbs = ['work','use','understand','grasp','struggle','surprise',
               'assign', 'explain','get','need','go','apply','order', 'like',
               'question','talk', 'ask', 'question','read','understanding',
               'turn', 'prepare','look','see', 'come','want','know','discuss',
               'continue','cover','seem']
extra_other = ['-PRON-','lot','etc','first','briefly','vs','also','pro','con','week',
               'specifically','forward','student','well',"n't",'past','towards',
               'mostly','little',"'s",'much','new','since','fair','jan','13th',
               '...','peer','primarily','properly','necessary','super','suck',
               'effectively','vs.','bit','quite','many','people','straight',
               'anything','hard','pretty','big','personally','--','really',
               'still','exactly','clearly','successfully','i.e','rough',
               'even','probably','way','pita','yet','especially','previously',
               'good','ok','extremely','mid','feb','-','basically','pleasantly',
               'surprising','old','person']
stop_words.extend(extra_verbs)
stop_words.extend(extra_other)
#dictionary of words to be replaced
sub_words = {'z score':'zscore', 'z-scores':'zscore',' repo ':'repository',' repos ':'repository', 'repositories':'repository',
             'data types':'datatype', 'data type':'datatype', 'open ended':'openended',' cs ':'computer science',
             'box plots':'boxplot', 'boxplots':'boxplot', 'box plot':'boxplot','barchartely':'barchart',
             'bar':'barchart', 'bar charts':'barchart', 'barplots':'barchart','bar plots':'barchart', 'barplot':'barchart',
             'pie charts':'piechart', 'bmi':'calculation', 'data cube':'datacube',
             'animal shelter':'dataset','salary data':'dataset','house elections':'dataset','campaign contributions':'dataset','campaign finance data set':'dataset', 'data sets':'dataset',
             'trend lines':'trendline', 'wayyy':'way','b/c':'because',
             'mini projects':'miniproject','mini project':'miniproject','mini-project':'miniproject',
             'co-lab':'colab', 'data camp':'datacamp', 'take-home':'takehome',
             'bc':'because','re run':'rerun', 'tiidy':'tidy', 'pallette':'palette', 'viz':'visualization','if-idf':'tf-idf',
             'data frames':'dataframe', 'data.frame':'dataframe', 'data frame':'dataframe',
             'babynames':'package', 'take home':'takehome', 'k means':'kmean',
             'professor baumer':'professor', 'professor clark':'professor',
             'pre-processing':'preprocessing', 'data set':'dataset', 'tf idf':'tfidf',
             'n/a':'nan','nothing':'nan','not really':'nan','nope':'nan','not particularly':'nan',
             'n.a':'nan','none':'nan','r studio':'rstudio'}

#variables used in future function calls
questions = ['topics','questions','struggles','surprises'] #list of question topics
roles = ['stu','ta','prof'] #list of roles
dfs = [stu,ta,prof] #list of dataframes

#process words, create dictionaries for future function calls
answers_question, answers_role = process_words(stop_words,sub_words,questions,dfs)#stop words definition
stop_words = list(stopwords.words('english'))
stop_words.extend(list(set(string.punctuation)))
extra_verbs = ['work','use','understand','grasp','struggle','surprise',
               'assign', 'explain','get','need','go','apply','order', 'like',
               'question','talk', 'ask', 'question','read','understanding',
               'turn', 'prepare','look','see', 'come','want','know','discuss',
               'continue','cover','seem']
extra_other = ['-PRON-','lot','etc','first','briefly','vs','also','pro','con','week',
               'specifically','forward','student','well',"n't",'past','towards',
               'mostly','little',"'s",'much','new','since','fair','jan','13th',
               '...','peer','primarily','properly','necessary','super','suck',
               'effectively','vs.','bit','quite','many','people','straight',
               'anything','hard','pretty','big','personally','--','really',
               'still','exactly','clearly','successfully','i.e','rough',
               'even','probably','way','pita','yet','especially','previously',
               'good','ok','extremely','mid','feb','-','basically','pleasantly',
               'surprising','old','person']
stop_words.extend(extra_verbs)
stop_words.extend(extra_other)
#dictionary of words to be replaced
sub_words = {'z score':'zscore', 'z-scores':'zscore',' repo ':'repository',' repos ':'repository', 'repositories':'repository',
             'data types':'datatype', 'data type':'datatype', 'open ended':'openended',' cs ':'computer science',
             'box plots':'boxplot', 'boxplots':'boxplot', 'box plot':'boxplot','barchartely':'barchart',
             'bar':'barchart', 'bar charts':'barchart', 'barplots':'barchart','bar plots':'barchart', 'barplot':'barchart',
             'pie charts':'piechart', 'bmi':'calculation', 'data cube':'datacube',
             'animal shelter':'dataset','salary data':'dataset','house elections':'dataset','campaign contributions':'dataset','campaign finance data set':'dataset', 'data sets':'dataset',
             'trend lines':'trendline', 'wayyy':'way','b/c':'because',
             'mini projects':'miniproject','mini project':'miniproject','mini-project':'miniproject',
             'co-lab':'colab', 'data camp':'datacamp', 'take-home':'takehome',
             'bc':'because','re run':'rerun', 'tiidy':'tidy', 'pallette':'palette', 'viz':'visualization','if-idf':'tf-idf',
             'data frames':'dataframe', 'data.frame':'dataframe', 'data frame':'dataframe',
             'babynames':'package', 'take home':'takehome', 'k means':'kmean',
             'professor baumer':'professor', 'professor clark':'professor',
             'pre-processing':'preprocessing', 'data set':'dataset', 'tf idf':'tfidf',
             'n/a':'nan','nothing':'nan','not really':'nan','nope':'nan','not particularly':'nan',
             'n.a':'nan','none':'nan','r studio':'rstudio'}

#variables used in future function calls
questions = ['topics','questions','struggles','surprises'] #list of question topics
roles = ['stu','ta','prof'] #list of roles
dfs = [stu,ta,prof] #list of dataframes

#process words, create dictionaries for future function calls
answers_question, answers_role = process_responses(stop_words,sub_words,questions,dfs)

#tfidf calculations
tfidf(answers_question,questions,roles) #tfidf calculation by question
tfidf(answers_role,roles,questions) #tfidf calculation by role

#create combined csv for roles
old = []
old.append(pd.read_csv(data_path+"stu.csv"))
old.append(pd.read_csv(data_path+"ta.csv"))
old.append(pd.read_csv(data_path+"prof.csv"))
combine_csv("all_roles.csv",old)
#create combined csv for questions
old = []
old.append(pd.read_csv(data_path+"topics.csv"))
old.append(pd.read_csv(data_path+"struggles.csv"))
old.append(pd.read_csv(data_path+"questions.csv"))
old.append(pd.read_csv(data_path+"surprises.csv"))
combine_csv("all_questions.csv",old)
