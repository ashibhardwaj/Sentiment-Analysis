
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Jan 14 14:45:11 2018

Title: Twitter Sentiment Analysis
@author: ashibhardwaj

"""


#importing essential libraries
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


#path of the dataset being used
path= '/Users/ashibhardwaj/Desktop/A/ML/Datasets/Tweets.csv'

#loading dataset
data= pd.read_csv(path)

#setting input variable
X= data.iloc[:, 10].values
#df = pd.DataFrame(X)

#setting output variable
y= data.iloc[:, 1].values

#splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#converting input text into bag of words

"""CountVectorizer produces a sparse representation of counts of tokens in each tweet.
It creates a 2D matrix with row corresponding to a tweet and columns corresponding to the
number of times a word (token/feature) occurs.
Here vect is an object of class CountVectorizer, which is used to convert collection of text 
documents to matrix of token counts."""

vect= CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = 0.75, min_df=3)

#Learns vocabulary dictionary of all tokens from all the tweets i.e. makes note of all the words present.
vect.fit(X_train)

#Gives token count for every instance so the matrix can be used to train the model.
X_train_dtm= vect.transform(X_train)
X_test_dtm= vect.transform(X_test)

#Building the model and making predictions.
NB= MultinomialNB()
NB.fit(X_train_dtm,y_train)
y_pred = NB.predict(X_test_dtm)

print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100, '%',sep=' ')
print('Confusion Matrix: ', metrics.confusion_matrix(y_test,y_pred), sep='\n')

