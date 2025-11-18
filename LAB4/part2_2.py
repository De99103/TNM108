from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=
True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)

print(twenty_test.target_names)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', MultinomialNB()),
])

text_clf.fit(twenty_train.data, twenty_train.target)

import numpy as np
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print("multinomialBC accuracy ",np.mean(predicted == twenty_test.target))

# training SVM classifier
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
,max_iter=5, tol=None)),
])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print("SVM accuracy ",np.mean(predicted == twenty_test.target))

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
target_names=twenty_test.target_names))

print(metrics.confusion_matrix(twenty_test.target, predicted))