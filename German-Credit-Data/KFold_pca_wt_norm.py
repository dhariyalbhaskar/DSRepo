#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 03:44:00 2016

@author: bhaskar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 01:45:07 2016

@author: bhaskar
"""

import csv 
import numpy as np
from sklearn import svm, linear_model, metrics,neighbors
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
csv_file_object=csv.reader(open('german.csv'))

data=[]
for row in csv_file_object:
    data.append(row)
    
    
data=np.array(data, dtype=np.float64)
new_data=data[:,:-1]


pca=PCA(n_components=20)
pca_X=pca.fit_transform(new_data)

#KFold NN-----------------------------------------------------

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

spec=tp.mean()/(tp.mean()+fn.mean())
sens=tn.mean()/(tn.mean()+fp.mean())

print("Spec",spec)
print("Sens",sens)
#NN----------------------------------------------------------
print("\nKFold Neural Nets")
X=new_data
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

spec=tp.mean()/(tp.mean()+fn.mean())
sens=tn.mean()/(tn.mean()+fp.mean())

print("Spec",spec)
print("Sens",sens)

#KFold NN-----------------------------------------------------

print("\nKFold Neural Nets")
X=new_data
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

spec=tp.mean()/(tp.mean()+fn.mean())
sens=tn.mean()/(tn.mean()+fp.mean())

print("Spec",spec)
print("Sens",sens)
#Kfold KNN---------------------------------------------------

print('\nKFold KNN')
X=new_data
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

spec=tp.mean()/(tp.mean()+fn.mean())
sens=tn.mean()/(tn.mean()+fp.mean())

print("Spec",spec)
print("Sens",sens)
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
spec=tp.mean()/(tp.mean()+fn.mean())
sens=tn.mean()/(tn.mean()+fp.mean())

print("Spec",spec)
print("Sens",sens)
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
    model = svm.SVC(kernel='linear', C=1, gamma='auto') 
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
spec=tp.mean()/(tp.mean()+fn.mean())
sens=tn.mean()/(tn.mean()+fp.mean())

print("Spec",spec)
print("Sens",sens)
#KFold SVM------------------------------------------------------
print('\nKFold SVM')
X=new_data
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
    model = svm.SVC(kernel='linear', C=1, gamma='auto') 
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
spec=tp.mean()/(tp.mean()+fn.mean())
sens=tn.mean()/(tn.mean()+fp.mean())

print("Spec",spec)
print("Sens",sens)
#KFOLD PCA+SVM-------------------------------------------
print('\nKFold PCA+SVM')
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
    model = svm.SVC(kernel='linear', C=1, gamma='auto') 
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
spec=tp.mean()/(tp.mean()+fn.mean())
sens=tn.mean()/(tn.mean()+fp.mean())

print("Spec",spec)
print("Sens",sens)
#FFold LR-------------------------------------------------------

print('\nKFold LR')
X=new_data
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
spec=tp.mean()/(tp.mean()+fn.mean())
sens=tn.mean()/(tn.mean()+fp.mean())

print("Spec",spec)
print("Sens",sens)
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
spec=tp.mean()/(tp.mean()+fn.mean())
sens=tn.mean()/(tn.mean()+fp.mean())

print("Spec",spec)
print("Sens",sens)