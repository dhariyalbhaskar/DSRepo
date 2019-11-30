#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 01:45:07 2016

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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import scipy
csv_file_object=csv.reader(open('german.csv'))
data=[]
for row in csv_file_object:
    data.append(row)
data=np.array(data, dtype=np.float64)
dataset=data[:,:-1]
target=data[:,-1]

#Feature Selection------------------------------------------------------
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 5 attributes
rfe = RFE(model,20) 
rfe = rfe.fit(dataset, target)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

#print(rfe.get_params)

col = dataset.shape[1];
row = dataset.shape[0];
print('----------------------------------');
kc=0
for i in range(0, col):
    if rfe.support_[i]==False:
        dataset = scipy.delete(dataset, i-kc, 1)
        kc=kc+1
    if rfe.support_[i]==True:
        print(i)
#print(dataset)

minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
normalized_data=minmax.fit_transform(dataset)
#print(normalized_data)

print('\nKFold PCA+SVM') 
pca=PCA(n_components=20)
pca_X=pca.fit_transform(normalized_data)

#KFold PCA+SVM----------------------------------------------
 
X=pca_X
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
    model = svm.SVC(kernel='rbf', C=1000, gamma=0.0001) 
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
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

#KFold SVM------------------------------------------------------
print('\nKFold SVM')
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
    model = svm.SVC(kernel='rbf', C=20, gamma=0.01) 
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
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


##KFold KNN--------------------------------------------------------------------
print('\nKFold KNN')
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
    knn_clf = neighbors.KNeighborsClassifier()
    knn_clf.fit(X_train,y_train)    
    y_pred= knn_clf.predict(X_test) 
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

#KFold PCA+KNN---------------------------------------------------------------
print('\nKFold PCA+KNN')
X=pca_X
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
    knn_clf = neighbors.KNeighborsClassifier()
    knn_clf.fit(X_train,y_train)    
    y_pred= knn_clf.predict(X_test) 
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
#KFold LR----------------------------------------------------------------------


print('\nKFold LR')
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
    model = linear_model.LogisticRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
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

#Kfold NN-----------------------------------------------------------------
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
    clf=MLPClassifier(solver='lbfgs',activation='logistic',alpha=1,hidden_layer_sizes=(5,5),learning_rate_init=0.75,random_state=1)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    acc=acc+[metrics.accuracy_score(y_test,y_pred)*100]
    tp=tp+[metrics.confusion_matrix(y_test,y_pred)[0][0]]
    tn=tn+[metrics.confusion_matrix(y_test,y_pred)[1][1]]
    fp=fp+[metrics.confusion_matrix(y_test,y_pred)[1][0]]
    fn=fn+[metrics.confusion_matrix(y_test,y_pred)[0][1]]
 
print("\nKFold NN")
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

#KFold PCA+LR

print("\nKFold PCA+LR")
X=pca_X
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
    model = linear_model.LogisticRegression(C=1e6)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
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

#Neural Nets-------------------------------------------------------------
print("\nKFold PCA+Neural Nets")
X=pca_X
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
    clf=MLPClassifier(solver='lbfgs',activation='logistic',alpha=1,hidden_layer_sizes=(5,5),learning_rate_init=0.75,random_state=1)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
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

'''
#KNearest Neighbour------------------------------------------------------------

print("\nKFold Nearest Neighbour")

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
    knn_clf = neighbors.KNeighborsClassifier()
    knn_clf.fit(X_train,y_train)    
    y_pred= knn_clf.predict(X_test)
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

'''
#Random Forest------------------------------------------------------------------
print("\nKFold Random Forest")

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
    model= RandomForestClassifier(n_estimators=1000)
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


#GradientBoost-----------------------------------------------------------------

print("\nKFold GB")

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
    clf = GradientBoostingClassifier(n_estimators=10, learning_rate=0.8, max_depth=3)
    clf.fit(X_train, y_train)
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

'''
#bagging Algorithms============================================================


print("\n BA:Random Forest")


X=normalized_data
y=data[:,-1]

acc=[]
kf=KFold(n_splits=3)
i=0
tp=[]
tn=[]
fp=[]
fn=[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model= RandomForestClassifier(n_estimators=80,max_features='auto')
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

'''
#bagging on NN

