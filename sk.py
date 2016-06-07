# -*- coding: utf-8 -*-
'''
Created on 2016年6月4日

@author: lvshaobo
'''

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
print twenty_train.target_names
count_vect = CountVectorizer()
print count_vect
X_train_counts = count_vect.fit_transform(twenty_train.data)
tfidf_transformer = TfidfTransformer()
print tfidf_transformer
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
docs_new = ['probability theory', 'long time no see']
X_new_counts = count_vect.transform(docs_new)
print X_new_counts
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
print predicted
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
    

import numpy as np
a = np.array([[261,  10,  12,  36], [  5, 380,   2,   2], [  7,  32, 353,   4], [  6,  11,   4, 377]]) 
print 261.0/a.sum(1)[0]
    
