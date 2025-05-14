import numpy as np
import pandas as pd
import gensim #This library has word2vec functionality
import os

from nltk import sent_tokenize
from gensim.utils import simple_preprocess
import nltk
nltk.download('punkt')

story = []
f = open('../001ssb.txt')
corpus = f.read()
raw_sent = sent_tokenize(corpus)
for sent in raw_sent:
    story.append(simple_preprocess(sent))
print(story)

model = gensim.models.Word2Vec(min_count = 2)
model.build_vocab(story)
model.train(story, total_examples=model.corpus_count, epochs=model.epochs)
print(model.wv.most_similar('daenerys'))
print(model.wv.similarity('arya', 'sansa'))
#Shape of the vectors
model.wv['deep'].shape
y = model.wv.index_to_key
#Getting all the vectors for the text corpus - this is what we need to pass to the model
vec = model.wv.get_normed_vectors()
print(vec)

#Principle Component Analysis - to reduce the dimension
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X = pca.fit_transform(model.wv.get_normed_vectors())
print(X)
print(X.shape)

#Plotting graph
import plotly.express as px
fig = px.scatter_3d(X[200:300], x=0, y=1, z=2, color=y[200:300])
fig.show()