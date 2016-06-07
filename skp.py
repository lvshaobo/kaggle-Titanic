# -*- coding: utf-8 -*-
'''
Created on 2016年6月6日

@author: lvshaobo
'''

from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

"""    
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
"""
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))])
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
"""
docs_test = ['probability theory', 'long time no see']
"""
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print predicted
"""
for doc, category in zip(docs_test, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
"""


# Results
print metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names)
print metrics.confusion_matrix(twenty_test.target, predicted)



from sklearn.grid_search import GridSearchCV

parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}           #引号内容不能变
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

twenty_train.target_names[gs_clf.predict(['God is love'])]
best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))