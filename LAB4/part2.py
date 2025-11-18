import sklearn
from sklearn.datasets import load_files

moviedir = '/Users/deemaabogheda/Desktop/study/Termin7/TNM108/LABS/LAB4/movie_reviews'

# loading all files. 
movie = load_files(moviedir, shuffle=True)
print(len(movie.data))
# target names ("classes") are automatically generated from subfolder names
print(movie.target_names)

# First file seems to be about a Schwarzenegger movie. 
print(movie.data[0][:500]) 

# first file is in "neg" folder
print(movie.filenames[0])

# first file is a negative review and is mapped to 0 index 'neg' in target_names
print(movie.target[0])

print("A detour: try out CountVectorizer & TF-IDF")

# import CountVectorizer, nltk
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Turn off pretty printing of jupyter notebook... it generates long lines
#%pprint

# Three tiny "documents"
docs = ['A rose is a rose is a rose is a rose.',
        'Oh, what a fine day it is.',
        "A day ain't over till it's truly over."]

# Initialize a CountVectorizer to use NLTK's tokenizer instead of its 
#    default one (which ignores punctuation and stopwords). 
# Minimum document frequency set to 1. 
fooVzer = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)

# .fit_transform does two things:
# (1) fit: adapts fooVzer to the supplied text data (rounds up top words into vector space) 
# (2) transform: creates and returns a count-vectorized output of docs
docs_counts = fooVzer.fit_transform(docs)

# fooVzer now contains vocab dictionary which maps unique words to indexes
print(fooVzer.vocabulary_)

# docs_counts has a dimension of 3 (document count) by 16 (# of unique words)
print(docs_counts.shape)

# this vector is small enough to view in a full, non-sparse form! 
print(docs_counts.toarray())

# Convert raw frequency counts into TF-IDF (Term Frequency -- Inverse Document Frequency) values
from sklearn.feature_extraction.text import TfidfTransformer
fooTfmer = TfidfTransformer()

# Again, fit and transform
docs_tfidf = fooTfmer.fit_transform(docs_counts)

# TF-IDF values
# raw counts have been normalized against document length, 
# terms that are found across many docs are weighted down ('a' vs. 'rose')
print(docs_tfidf.toarray())

# A list of new documents
newdocs = ["I have a rose and a lily.", "What a beautiful day."]

# This time, no fitting needed: transform the new docs into count-vectorized form
# Unseen words ('lily', 'beautiful', 'have', etc.) are ignored
newdocs_counts = fooVzer.transform(newdocs)
print(newdocs_counts.toarray())

# Split data into training and test sets
from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target, 
                                                          test_size = 0.20, random_state = 12)

# initialize CountVectorizer
movieVzer= CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000) # use top 3000 words only. 78.25% acc.
movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)         # use all 25K words. Higher accuracy

# fit and tranform using training text 
docs_train_counts = movieVzer.fit_transform(docs_train)

# 'screen' is found in the corpus, mapped to index 2290
print(movieVzer.vocabulary_.get('screen'))

# Likewise, Mr. Steven Seagal is present...
print(movieVzer.vocabulary_.get('seagal'))

# huge dimensions! 1,600 documents, 3K unique terms. 
print(docs_train_counts.shape)

# Convert raw frequency counts into TF-IDF values
movieTfmer = TfidfTransformer()
docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)
print(docs_train_tfidf.shape)  # still the same shape


# Using the fitted vectorizer and transformer, tranform the test data
docs_test_counts = movieVzer.transform(docs_test)
docs_test_tfidf = movieTfmer.transform(docs_test_counts)

print("Training and testing a Naive Bayes classifier" )
# Now ready to build a classifier. 
# We will use Multinominal Naive Bayes as our model
from sklearn.naive_bayes import MultinomialNB
# Train a Multimoda Naive Bayes classifier. Again, we call it "fitting"
clf = MultinomialNB()
print(clf.fit(docs_train_tfidf, y_train))

# Predict the Test set results, find accuracy
y_pred = clf.predict(docs_test_tfidf)
print(sklearn.metrics.accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("fake review test:")

# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride', 
            'Steven Seagal was terrible', 'Steven Seagal shone through.', 
              'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through', 
              "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough', 
              'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

reviews_new_counts = movieVzer.transform(reviews_new)         # turn text into count vector
reviews_new_tfidf = movieTfmer.transform(reviews_new_counts)  # turn into tfidf vector

# have classifier make a prediction
pred = clf.predict(reviews_new_tfidf)
# print out results
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))


print("Pipeline + GridSearch on movie reviews")
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score
#pipeline : combines two steps: TF_IDF vectorizer and Multinomial Naive Bayes classifier
#where tidf vectorizer turns raw movie reviews into TF-IDF features
# and MultinomialNB does the classification based on those features positive/negative. 
#define pipeline : TF_IDF + Naive Bayes
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=nltk.word_tokenize)),
    ('clf', MultinomialNB()),
])

# define parameter grid for GridSearch 
parameters = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': [True, False],
    'clf__alpha': [1.0, 0.1, 0.01],
} # 2×2×3 = 12 total models to test.

# 3. grid search using it to find best parameters to TF-IDF and Naive Bayes
gs_clf = GridSearchCV(text_clf, parameters, cv=3, n_jobs=-1) # n_jobs=-1 uses all available cores
#cv : number of folds for cross-validation.
gs_clf.fit(docs_train, y_train)

print("Best parameters:", gs_clf.best_params_)
print("Best CV score:", gs_clf.best_score_)

# 4. evaluate on test set
y_pred_gs = gs_clf.predict(docs_test)
print("Test accuracy with best params:", accuracy_score(y_test, y_pred_gs))
