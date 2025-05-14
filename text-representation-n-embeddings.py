import numpy as np
import pandas as pd

#Bag of Words Implementation
df = pd.DataFrame({"text":["people watch people",
                         "people watch movie",
                         "people write movie",
                          "movie write movie"],"output":[1,1,0,0]})
print(df)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() #This method is used for Bag of Words technique

bow = cv.fit_transform(df['text'])
#vocabulary - this returns indexes the words
print(cv.vocabulary_)
#Converting the text dataframe to matrix based on above indexes and frequency of the words
print(bow.toarray())
#Print each row 
print(bow[0].toarray())
#Trying this with new sentence
print(cv.transform(['Bappy watch dswithbappy']).toarray())
#Creating input and output for the model
X = bow.toarray()
y = df['output']


#N-gram
df_n = pd.DataFrame({"text":["people watch people",
                         "people watch movie",
                         "people write movie",
                          "movie write movie"],"output":[1,1,0,0]})
print(df_n)
#Bi-grams
from sklearn.feature_extraction.text import CountVectorizer
cv_n = CountVectorizer(ngram_range=(2,2))
bow_n = cv_n.fit_transform(df['text'])
print(cv_n.vocabulary_)
#Tri-grams
from sklearn.feature_extraction.text import CountVectorizer
cv_3 = CountVectorizer(ngram_range=(3,3))
bow_3 = cv_3.fit_transform(df_n['text'])
print(cv_3.vocabulary_)


#TF-IDF: Term Frequency - Inverse Document Frequency
df_tf = pd.DataFrame({"text":["people watch people",
                         "people watch movie",
                         "people write movie",
                          "movie write movie"],"output":[1,1,0,0]})
print(df_tf)
from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer()
arr =  tfid.fit_transform(df['text']).toarray()
print(arr)
print(tfid.idf_)