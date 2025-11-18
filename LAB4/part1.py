d1 = "The sky is blue."
d2 = "The sun is bright."
d3 = "The sun in the sky is bright."
d4 = "We can see the shining sun, the bright sun."
Z = (d1,d2,d3,d4)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer ()

print(vectorizer)
my_stop_words={"the","is"}
my_vocabulary={'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}

vectorizer=CountVectorizer(stop_words=my_stop_words,vocabulary=my_vocabulary)

print(vectorizer.vocabulary)
print(vectorizer.stop_words)

print ("--"*20)
smatrix = vectorizer.transform(Z)
print(smatrix)

print("todense()")
matrix = smatrix.todense()
print(matrix)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(norm="l2")
tfidf_transformer.fit(smatrix)

# print idf values
feature_names = vectorizer.get_feature_names_out()

import pandas as pd
df_idf=pd.DataFrame(tfidf_transformer.idf_, index=feature_names,columns=["idf_weights"])
# sort ascending
df_idf.sort_values(by=['idf_weights'])
print(df_idf)

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(smatrix)

# get tfidf vector for first document
first_document = tf_idf_vector[0] # first document "The sky is blue."
# print the scores
df=pd.DataFrame(first_document.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)

print(df)

print("Document Similarity using TF-IDF")
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
print(tfidf_matrix.shape)
#output: (4, 11) : This means that we have created a TF-IDF matrix consisting of 4 rows (since we have 4 documents)
# and 11 column

from sklearn.metrics.pairwise import cosine_similarity
cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
print(cos_similarity)  #output: [[1.         0.28867513 0.57735027 0.40824829]] (4 values (one for each document))
# The first value of the array is 1.0 since it is the cosine similarity between the first document with itself.

# Take the cos similarity of the third document (cos similarity=0.52)
import math
third_sim = cos_similarity[0, 2]   # similarity between doc1 and doc3
angle_in_radians = math.acos(third_sim)
print(math.degrees(angle_in_radians)) # This is the angle between the first and the third document of our document set.

print("Classifying Text")

from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
print(data.target_names)

my_categories = ['rec.sport.baseball','rec.motorcycles','sci.space','comp.graphics'] # here we select only 4 categories of 20 available
train = fetch_20newsgroups(subset='train', categories=my_categories) # 
test = fetch_20newsgroups(subset='test', categories=my_categories)

print("train.data")
print(len(train.data))

print("test.data")
print(len(test.data))

print(train.data[9]) # print 10th document in training set

from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer()  
#CountVectorizer converts the raw text into a bag-of-words matrix. 
#Each document becomes a row.
#Each word in the vocabulary becomes a column.
#Each cell contains the word count (term frequency).
#This is the first step toward TF-IDF.


X_train_counts=cv.fit_transform(train.data)  
# makes the vectorizer learn the vocabulary from the training documents (fit) and then convert 
# #every training document into a bag-of-words count vector (transform) 
# -> sparse matrix of shape (n_samples, n_features)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer() # TfidfTransformer takes the word-count matrix and converts it into a TF-IDF matrix.

X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts) # fit_transform means that it first learns the IDF vector (fit)
# and then transforms the count matrix to a TF-IDF representation (transform).
#This applies the IDF weighting to reduce the influence of common words.
#X_train_tfidf is the final feature matrix we will train the classifier on.

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train_tfidf, train.target)

#This creates and trains a Multinomial Naive Bayes classifier.
#It learns which TF-IDF features are typical for each text category.
#X_train_tfidf is the TF-IDF matrix of the training documents.
#train.target is the list of class labels for each document.
#After this line. the classifier has learned patterns in the text.

docs_new = ['Pierangelo is a really good baseball player','Maria rides her motorcycle', 'OpenGL on the GPU is fast', 'Pierangelo rides his motorcycle and goes to play football since he is a good football player too.']
X_new_counts = cv.transform(docs_new) # here again we use the cv.transform method to into word count vectors. 
X_new_tfidf = tfidf_transformer.transform(X_new_counts) # then we use the tfidf_transformer.transform method to convert the count vectors into TF-IDF vectors using the IDF vector learned earlier.
#The new documents become TF-IDF vectors so they match the format expected by the classifier.

predicted = model.predict(X_new_tfidf) # uses the trained Naive Bayes model to guess the category of each new document.
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, train.target_names[category]))