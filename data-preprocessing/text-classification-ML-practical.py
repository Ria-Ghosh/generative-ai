import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 255)

dataPath = '../IMDB Dataset.csv'

df = pd.read_csv(dataPath)
print(df.head())
print(df.shape)

#Sampling data for time efficiency
df = df.iloc[:10000]
print(df.head())
print(df.shape)

#Sentiment value count
print(df['sentiment'].value_counts())

#Check if there is any null data present
print(df.isnull().sum())

#Check to look for duplicates
print(df.duplicated().sum())
#Drop the duplicates
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())

#Basic Preprocessing
"""
1. Remove tags - HTML
2. Lower case
3. Remove Stopwords
"""
import re
def removeTags(rawText):
    cleanedText = re.sub(re.compile('<.*?>'), '', rawText)
    return cleanedText
df['review'] = df['review'].apply(removeTags)
print(df.head())

df['review'] = df['review'].apply(lambda x:x.lower())
print(df['review'][0])

swList = stopwords.words('english')
df['review'] = df['review'].apply(lambda x: [item for item in x.split() if item not in swList]).apply(lambda x: " ".join(x))
print(df['review'][0])

#Assigning input and output values of the dataset
X = df.iloc[:, 0:1]
y = df['sentiment']
print(X.head(), y.head())

#Since output is in string, we need to convert that to numeric values - using label encoders
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)

#Perform train/test split
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=1)
print(XTrain.shape, XTest.shape)

#Applying BoW
cv = CountVectorizer()
XTrainBow = cv.fit_transform(XTrain['review']).toarray()
XTestBow = cv.transform(XTest['review']).toarray()
print(XTrainBow, XTestBow)

#Training data on Naive Bayes ML Model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(XTrainBow, yTrain)
yPred = gnb.predict(XTestBow)
print(accuracy_score(yTest, yPred))
print(confusion_matrix(yTest, yPred))

#Training data on Random Forest Classifier ML Model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(XTrainBow, yTrain)
yPred = rf.predict(XTestBow)
print(accuracy_score(yTest, yPred))

#Reapplying BoW by limiting number of features instead of taking all features which decreases model complexity
cv = CountVectorizer(max_features=3000)
XTrainBow = cv.fit_transform(XTrain['review']).toarray()
XTestBow = cv.transform(XTest['review']).toarray()
rf = RandomForestClassifier()
rf.fit(XTrainBow, yTrain)
yPred = rf.predict(XTestBow)
print(accuracy_score(yTest, yPred))

#N grams
cv = CountVectorizer(ngram_range=(1,2), max_features=5000)
XTrainBow = cv.fit_transform(XTrain['review']).toarray()
XTestBow = cv.transform(XTest['review']).toarray()
rf = RandomForestClassifier()
rf.fit(XTrainBow, yTrain)
yPred = rf.predict(XTestBow)
print(f'N-grams accuracy:{accuracy_score(yTest, yPred)}')

#TF-IDF
tfidf = TfidfVectorizer()
XTrainTfidf = tfidf.fit_transform(XTrain['review']).toarray()
XTestTfidf = tfidf.transform(XTest['review'])
rf = RandomForestClassifier()
rf.fit(XTrainTfidf, yTrain)
yPred = rf.predict(XTestTfidf)
print(f'TF-IDF Accuracy: {accuracy_score(yTest, yPred)}')