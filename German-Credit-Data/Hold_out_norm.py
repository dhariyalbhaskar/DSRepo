#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 01:21:20 2016

@author: bhaskar
"""

import csv 
import numpy as np
from sklearn import preprocessing, svm, linear_model, metrics,cross_validation,neighbors
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
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
X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

model = svm.SVC(kernel='linear', C=1, gamma='auto')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

e_ar=[]
e_ar=e_ar + [y_pred]
        
print("\n\n ********** RESULTS BY NORMALISED SVM ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

tp=metrics.confusion_matrix(y_test,y_pred)[0][0]
tn=metrics.confusion_matrix(y_test,y_pred)[1][1]
fp=metrics.confusion_matrix(y_test,y_pred)[1][0]
fn=metrics.confusion_matrix(y_test,y_pred)[0][1]

spec=tp/(tp+fn)
sens=tn/(tn+fp)

print("Spec",spec)
print("Sens",sens)


#LR---------------------------------------------------------------------------------

X=normalized_data
y=data[:,-1]

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

logreg=linear_model.LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

e_ar=e_ar+  [y_pred]
print("\n\n ********** RESULTS BY Normalized LR ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))
tp=metrics.confusion_matrix(y_test,y_pred)[0][0]
tn=metrics.confusion_matrix(y_test,y_pred)[1][1]
fp=metrics.confusion_matrix(y_test,y_pred)[1][0]
fn=metrics.confusion_matrix(y_test,y_pred)[0][1]

spec=tp/(tp+fn)
sens=tn/(tn+fp)

print("Spec",spec)
print("Sens",sens)

#KNN------------------------------------------------------------------------------

X=normalized_data
y=data[:,-1]

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

knn_clf = neighbors.KNeighborsClassifier()
knn_clf.fit(X_train,y_train)    
y_pred= knn_clf.predict(X_test)

e_ar=e_ar+  [y_pred]
print("\n\n ********** RESULTS BY Normalized KNN ********** \n")

print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

tp=metrics.confusion_matrix(y_test,y_pred)[0][0]
tn=metrics.confusion_matrix(y_test,y_pred)[1][1]
fp=metrics.confusion_matrix(y_test,y_pred)[1][0]
fn=metrics.confusion_matrix(y_test,y_pred)[0][1]

spec=tp/(tp+fn)
sens=tn/(tn+fp)

print("Spec",spec)
print("Sens",sens)
#NN---------------------------------------------------------------

X=normalized_data
y=data[:,-1]

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

clf=MLPClassifier(solver='lbfgs',activation='logistic',alpha=1,hidden_layer_sizes=(5,5),learning_rate_init=0.75,random_state=0)
clf.fit(X_train,y_train)


y_pred=clf.predict(X_test)

e_ar=e_ar+ [y_pred]
print("\n\n ********** RESULTS BY PCA+MLP ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))
tp=metrics.confusion_matrix(y_test,y_pred)[0][0]
tn=metrics.confusion_matrix(y_test,y_pred)[1][1]
fp=metrics.confusion_matrix(y_test,y_pred)[1][0]
fn=metrics.confusion_matrix(y_test,y_pred)[0][1]

spec=tp/(tp+fn)
sens=tn/(tn+fp)

print("Spec",spec)
print("Sens",sens)

#PCA+SVM-------------------------------------------------------

pca=PCA(n_components=20)
pca_X=pca.fit_transform(normalized_data)

X=pca_X
y=data[:,-1]

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

model = svm.SVC(kernel='rbf', C=10, gamma='auto') 
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

e_ar=e_ar+  [y_pred]
print("\n\n ********** RESULTS BY PCA+SVM ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

tp=metrics.confusion_matrix(y_test,y_pred)[0][0]
tn=metrics.confusion_matrix(y_test,y_pred)[1][1]
fp=metrics.confusion_matrix(y_test,y_pred)[1][0]
fn=metrics.confusion_matrix(y_test,y_pred)[0][1]

spec=tp/(tp+fn)
sens=tn/(tn+fp)

print("Spec",spec)
print("Sens",sens)
#PCA+LR-------------------------------------------------------------

pca=PCA(n_components=20)
pca_X=pca.fit_transform(normalized_data)

X=pca_X
y=data[:,-1]

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

logreg=linear_model.LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

e_ar=e_ar+  [y_pred]
print("\n\n ********** RESULTS BY PCA+LR ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

tp=metrics.confusion_matrix(y_test,y_pred)[0][0]
tn=metrics.confusion_matrix(y_test,y_pred)[1][1]
fp=metrics.confusion_matrix(y_test,y_pred)[1][0]
fn=metrics.confusion_matrix(y_test,y_pred)[0][1]

spec=tp/(tp+fn)
sens=tn/(tn+fp)

print("Spec",spec)
print("Sens",sens)

#PCA+KNN--------------------------------------------------------------------------

X=pca_X
y=data[:,-1]

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

knn_clf = neighbors.KNeighborsClassifier()
knn_clf.fit(X_train,y_train)    
y_pred= knn_clf.predict(X_test)

e_ar=e_ar+  [y_pred]
print("\n\n ********** RESULTS BY PCA+KNN ********** \n")

print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

tp=metrics.confusion_matrix(y_test,y_pred)[0][0]
tn=metrics.confusion_matrix(y_test,y_pred)[1][1]
fp=metrics.confusion_matrix(y_test,y_pred)[1][0]
fn=metrics.confusion_matrix(y_test,y_pred)[0][1]

spec=tp/(tp+fn)
sens=tn/(tn+fp)

print("Spec",spec)
print("Sens",sens)

#PCA+NN--------------------------------------------------------------------------------
X=pca_X
y=data[:,-1]

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

clf=MLPClassifier(solver='lbfgs',activation='logistic',alpha=1,hidden_layer_sizes=(5,5),learning_rate_init=0.75,random_state=0)
clf.fit(X_train,y_train)


y_pred=clf.predict(X_test)

e_ar=e_ar+ [y_pred]
print("\n\n ********** RESULTS BY PCA+MLP ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))
tp=metrics.confusion_matrix(y_test,y_pred)[0][0]
tn=metrics.confusion_matrix(y_test,y_pred)[1][1]
fp=metrics.confusion_matrix(y_test,y_pred)[1][0]
fn=metrics.confusion_matrix(y_test,y_pred)[0][1]

spec=tp/(tp+fn)
sens=tn/(tn+fp)

print("Spec",spec)
print("Sens",sens)
'''
#Ensembling--------------------------------------------------------------------

print("\nEnsembling!!!")
e_ar=np.array(e_ar)

pred=np.zeros((300,1))

count1=0
count2=0
i=0
j=0
k=0

for i in range(0,len(e_ar[0])):           #300
    for j in range(0,len(e_ar)):
        if e_ar[j][i]==1:
            count1=count1+1
        else:
            count2=count2+1
    if count1>count2:
        pred[i]=1
    else:
        pred[i]=2
    count1=0
    count2=0
    k=k+1
    
i=0
pos,neg=0,0 
for i in range(0,len(pred)):
    if pred[i] == y_test[i]:
        pos = pos+1
    else:
        neg = neg+1
        
print("\n\n ********************* RESULTS ************ \n")

print("correctly classified     --> ",pos)
print("not correctly classified -->",neg)
print("percentage(%) accuracy   --> ",(float(pos)/float(len(pred)))*100)


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