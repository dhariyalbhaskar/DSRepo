#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 12:28:21 2016

@author: bhaskar
"""
import csv 
import numpy as np
from sklearn import preprocessing, svm, linear_model, metrics,neighbors
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn import cross_validation

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
normalized_data=minmax.fit_transform(new_data)
print("\nBA: Neural Nets")
X=normalized_data
y=data[:,-1]

acc=[]
kf=KFold(n_splits=10)
i=0
tp=[]
tn=[]
fp=[]
fn=[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf=MLPClassifier(solver='lbfgs',activation='logistic',alpha=1,hidden_layer_sizes=(5,5),learning_rate_init=0.75,random_state=7)
    model = BaggingClassifier(base_estimator=clf,n_estimators=10,max_features=24)
    model.fit(X_train, y_train)
    y_pred= model.predict(X_test)
    acc=acc+[metrics.accuracy_score(y_test,y_pred)*100]
    tp=tp+[metrics.confusion_matrix(y_test,y_pred)[0][0]]
    tn=tn+[metrics.confusion_matrix(y_test,y_pred)[1][1]]
    fp=fp+[metrics.confusion_matrix(y_test,y_pred)[1][0]]
    fn=fn+[metrics.confusion_matrix(y_test,y_pred)[0][1]]
    
acc=np.array(acc)
tp=np.array(tp)
tn=np.array(tn)
fp=np.array(fp)
fn=np.array(fn)
print("Accuracy",acc.mean())
print('tp',tp.mean())
print('tn',tn.mean())
print('fp',fp.mean())
print('fn',fn.mean())
