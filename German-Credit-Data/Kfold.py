# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:43:00 2016

@author: bhaskar
"""

import csv 
import numpy as np
from sklearn import preprocessing, svm, linear_model, metrics
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
csv_file_object=csv.reader(open('german.csv'))

data=[]
for row in csv_file_object:
    data.append(row)
    
    
data=np.array(data, dtype=np.float64)
new_data=data[:,:-1]

minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
normalized_data=minmax.fit_transform(new_data)
print('Nomalized Data')
print(normalized_data)


#kf = KFold(normalized_data.shape[0],n_folds=10,shuffle=False)
print('Principle component analysis') 
pca=PCA(n_components=20)
pca_X=pca.fit_transform(normalized_data)
#print(pca_X)

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
print(acc.mean())
print('tp',tp.mean())
print('tn',tn.mean())
print('fp',fp.mean())
print('fn',fn.mean())

'''
model = svm.SVC(kernel='rbf', C=6, gamma=1/14) 
scores=cross_val_score(model,X,y,cv=10,scoring='accuracy')
print(scores)
'''
