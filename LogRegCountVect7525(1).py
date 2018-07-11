#!/usr/bin/env python

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

import csv
import sys
import nltk

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

trainRatio=0.75

df=pd.read_csv('/home/fangran/gensim/submission/rawLabels.csv', names=["new_reviews","senti_list"])

#randomize rows
df=shuffle(df)

#if its not labelled correctly (pos,neg,neut not spelled correctly), remove row
df1 = df.loc[(df['senti_list'] == 'neg') | (df['senti_list'] == 'neut') | (df['senti_list'] == 'pos')]
print(df1)

#clean data
#cleaned_sent=[]
#for index,row in df.iterrows():
#    text=row["new_reviews"]

df1.to_csv('/home/fangran/gensim/submission/shuffled.csv',header=False,index=False, index_label=False)

df=pd.read_csv('/home/fangran/gensim/submission/shuffled.csv')

rowNum=len(df.index)
#returns floor
trainNum=int(trainRatio*rowNum)
testNum=rowNum-trainNum

training=df[:trainNum]
testing=df[trainNum:]

print("Number of training sentences: ")
print(trainNum)
print("Number of testing sentences: ")
print(testNum)

training.to_csv('/home/fangran/gensim/submission/LogRTraining_7525.csv',header=False,index=False, index_label=False)
testing.to_csv('/home/fangran/gensim/submission/LogRTesting_7525.csv',header=False,index=False, index_label=False)

simpletext = []
sometext = []
trainLabs=[]

vectorizer = CountVectorizer(stop_words='english')
#vectorizer = CountVectorizer(min_n=1, max_n=2)

with open('/home/fangran/gensim/submission/LogRTraining_7525.csv','r') as simple:
    sometext = csv.reader(simple)
    for row in sometext:
    # print 'row1 ', row
        #it wants a list of strings, not one string.
        #train_features=vectorizer.fit_transform(row[0])
        #tblb = ()
        #tblb = (row[0],row[1])
        simpletext.append(row[0])
        #trainlabs.append(row[1])
        if (row[1] == "pos"):
            trainLabs.append(int(1))
        elif (row[1] == "neut"):
            trainLabs.append(int(0))
        elif (row[1] == "neg"):
            trainLabs.append(int(-1))

    train_features=vectorizer.fit_transform(simpletext)

testworthy = []
testable = []
testLabs=[]

with open('/home/fangran/gensim/submission/LogRTesting_7525.csv','r') as testthis:
    testable = csv.reader(testthis)
    rowNum=0
    for row in testable:
    # print 'testable ', row
        #test_features=vectorizer.fit_transform(row[0])
        #tblb = ()
        #tblb = (row[0],row[1])
        #testworthy.append(tblb)
        testworthy.append(row[0])
        rowNum=rowNum+1
        if (row[1] == "pos"):
            testLabs.append(int(1))
        elif (row[1] == "neut"):
            testLabs.append(int(0))
        elif (row[1] == "neg"):
            testLabs.append(int(-1))

    test_features=vectorizer.transform(testworthy)

clf=LogisticRegression()
clf.fit(train_features, trainLabs)

predictions = clf.predict(test_features)

print(np.mean(predictions == testLabs))
fpr, tpr, thresholds = metrics.roc_curve(testLabs, predictions, pos_label=1)
print("LogisticRegression AUC: {0}".format(metrics.auc(fpr, tpr)))

print(metrics.classification_report(testLabs, predictions,digits=2))
