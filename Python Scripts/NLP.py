import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

os.chdir('/Users/ritik.shah/Desktop/Personal/Data Science/Machine Learning A-Z Template Folder/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing')

#we use the tsv file for NLP because csv(comma seperated files) have comma as a delimiter and comma can be a part of the text that would be analyzed
#tsv files are tab separated files
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting =3)

#cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#removing all other characters apart from a-z or A-Z
#once a character is removed it it replaced by a ' '
#converting all the upper case characters into lower case
#removing the non significant words (the, on, in, are, this, that)
#first we split the sentence into different words and then remove the stop words from it
#only keeping the root of the word and not the future, past or continuous tense of the word(loved->love)
#this is so that we can reduce the size of our sparse matrix
#this processing is called stemming
N=1000
corpus=[]
for i in range(0,N):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) 
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

#create a Bag of Words model
#taking all the unique words from the sentences and creating one column for each word
#we will get a matrix with lot of 0's and a matrix containing lot of 0's is called a sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
