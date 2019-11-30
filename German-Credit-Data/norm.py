#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 01:21:20 2016

@author: bhaskar
"""

import csv 
import numpy as np
from sklearn import preprocessing, svm, linear_model, metrics,cross_validation
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


csv_file_object=csv.reader(open('german.csv'))

data=[]
for row in csv_file_object:
    data.append(row)
    
    
data=np.array(data, dtype=np.float64)
new_data=data[:,:-1]    #this is training data
new_data=np.array(new_data, dtype=np.float64)

print("---------------With Normalization------------------")
#Data tranformation using MinMaxScaler
minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
normalized_data=minmax.fit_transform(new_data)      #normalizing training data

X=normalized_data
y=data[:,-1]
X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)

model = svm.SVC(kernel='linear', C=1, gamma='auto')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
        
print("\n\n ********** RESULTS BY NORMALISED SVM ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

#Logistic regression
logreg=linear_model.LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

print("\n\n ********** RESULTS BY NORMALISED LR ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

#PCA+SVM-------------------------------------------------------

pca=PCA(n_components=20)
pca_X=pca.fit_transform(normalized_data)

X=pca_X
y=data[:,-1]

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)

model = svm.SVC(kernel='rbf', C=10, gamma='auto') 
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("\n\n ********** RESULTS BY SVM+PCA ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))



#PCA+LR-------------------------------------------------------------

pca=PCA(n_components=20)
pca_X=pca.fit_transform(normalized_data)

X=pca_X
y=data[:,-1]

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

logreg=linear_model.LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

print("\n\n ********** RESULTS BY PCA+LR ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

#LR---------------------------------------------------------------------------------

X=normalized_data
y=data[:,-1]

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

logreg=linear_model.LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

print("\n\n ********** RESULTS BY Normalized LR ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

X=normalized_data
y=data[:,-1]
X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.2,random_state=1)
model= RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
y_pred= model.predict(X_test)

print("\n\n ********** RESULTS BY RF ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))


