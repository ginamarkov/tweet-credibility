from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import pandas as pd
import numpy as np

stop = pd.read_csv("onix.csv", names=['word'])
stoplist = stop['word'].tolist()

#count_vect = CountVectorizer(stop_words=stoplist, analyzer='word')

liar=pd.read_csv("LIAR-PLUS-master/dataset/train2.tsv", sep='\t')

liar.columns = ['drop', 'ID', 'label', 'statement', 'subject', 'speaker', 'speaker title', 'state', 'party', 'barely_count', 
'false_count', 'half_count', 'mostly_count', 'fire_count', 'location', 'context']
x_liar = liar['statement']


liar2=pd.read_csv("LIAR-PLUS-master/dataset/test2.tsv", sep='\t')
liar2.columns = ['drop', 'ID', 'label', 'statement', 'subject', 'speaker', 'speaker title', 'state', 'party', 'barely_count', 'false_count', 'half_count', 'mostly_count', 'fire_count', 'location', 'context']
x_liar2 = liar2['statement']

def int_decode(label):
	if label == 'true':
		label=0
	elif label == 'half-true':
		label=1
	elif label == 'barely-true':
		label=2
	elif label == 'mostly-true':
		label=3
	elif label == 'false':
		label=4
	elif label == 'pants-fire':
		label=5
	return label


label_new = []
label_old = liar['label']
for i in range(0, len(liar)):
	label_new.append(int_decode(label_old[i]))

label_new2 = []
label_old2 = liar2['label']
for i in range(0, len(liar2)):
	label_new2.append(int_decode(label_old2[i]))


from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

text_clf = Pipeline([
	('vect', CountVectorizer(stop_words=stoplist, analyzer='char_wb', ngram_range=(2, 2))),
	('tfidf', TfidfTransformer()),
	('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))
	])

text_clf.fit(x_liar, label_new)
predicted = text_clf.predict(x_liar2)
print(np.mean(predicted == label_new2))


text_clf = Pipeline([
	('vect', CountVectorizer(stop_words=stoplist, analyzer='word')),
	('tfidf', TfidfTransformer()),
	('clf', MultinomialNB())
	])

text_clf.fit(x_liar, label_new)
predicted = text_clf.predict(x_liar2)
print(np.mean(predicted == label_new2))


text_clf = Pipeline([
	('vect', CountVectorizer(stop_words=stoplist, analyzer='word')),
	('tfidf', TfidfTransformer()),
	('clf', KNeighborsClassifier())
	])

text_clf.fit(x_liar, label_new)
predicted = text_clf.predict(x_liar2)
print(np.mean(predicted == label_new2))

text_clf = Pipeline([
	('vect', CountVectorizer(stop_words=stoplist, analyzer='word')),
	('tfidf', TfidfTransformer()),
	('clf', LinearSVC())
	])

text_clf.fit(x_liar, label_new)
predicted = text_clf.predict(x_liar2)
print(np.mean(predicted == label_new2))

import seaborn as sns


import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=stoplist)
features = tfidf.fit_transform(x_liar).toarray()
labels = label_new

'''
from sklearn.model_selection import cross_val_score
models = [
    SVC(),
    MultinomialNB(),
    KNeighborsClassifier(),
    SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    print('hi')
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    print(accuracies)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
print(cv_df)
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
'''
