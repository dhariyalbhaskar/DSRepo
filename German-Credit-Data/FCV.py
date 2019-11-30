# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 17:52:01 2016

@author: bhaskar
"""

import csv 
import numpy as np
from sklearn import preprocessing, svm, linear_model, metrics
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA

#from sklearn import preprocessing as prep
#--------fetching data
csv_file_object=csv.reader(open('german.csv'))
#header=next(csv_file_object)


#--------------------------------
data=[]
for row in csv_file_object:
    data.append(row)
    
    
data=np.array(data, dtype=np.float64)
new_data=data[:,:-1]

#Data tranformation using MinMaxScaler
minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
normalized_data=minmax.fit_transform(new_data)
print('Nomalized Data')
print(normalized_data)

#Data transformation using StandardScaler
standard=preprocessing.StandardScaler()
stndardised_data=standard.fit_transform(new_data)
print('Standardised Data')
print(stndardised_data)

print('Principle component analysis') 
pca=PCA(n_components=20)
pca_X=pca.fit_transform(normalized_data)
print(pca_X)

var1=np.cumsum(np.round(pca.explained_variance_, decimals=4)*100)
print('PCA Cum Var')
print(var1)

#SVM  & PCA
X=pca_X
y=data[:,-1]

#train_x,test_x,train_y,test_y=train_test_split(X,y,random_state=10)

model = svm.SVC(kernel='linear', C=1, gamma=1) 
new=cross_val


print(new)
pos,neg=0,0 
for i in range(0,len(new)):
    if new[i] == test_y[i]:
        pos = pos+1
    else:
        neg = neg+1

print("\n\n ********** RESULTS BY SVM+PCA ********** \n")

print("correctly classified     --> ",pos)
print("not correctly classified -->",neg)
print("percentage(%) accuracy   --> ",(float(pos)/float(len(new)))*100)

print(metrics.confusion_matrix(test_y,new))

#SVM only
X=normalized_data[:,:-1]
y=data[:,-1]

#train_x,test_x,train_y,test_y=train_test_split(X,y,random_state=2)

model = svm.SVC(kernel='linear', C=1, gamma=1) 
model.fit(train_x,train_y)

new = list(model.predict(test_x))
#print(new)
pos,neg=0,0 
for i in range(0,len(new)):
    if new[i] == test_y[i]:
        pos = pos+1
    else:
        neg = neg+1
        
print("\n\n ********** RESULTS BY SVM ********** \n")

print("correctly classified     --> ",pos)
print("not correctly classified -->",neg)
print("percentage(%) accuracy   --> ",(float(pos)/float(len(new)))*100)
print(metrics.confusion_matrix(test_y,new))
'''
#Logistic regression
train_x=pca_X[:800,:-1]
train_x=np.array(train_x)
train_y=data[:800,-1]

test_x=pca_X[800:,:-1]
test_x=np.array(test_x)

test_y=data[800:,-1]

logreg=linear_model.LogisticRegression()
logreg.fit(train_x,train_y)
y_pred=logreg.predict(test_x)
new = list(y_pred)

pos,neg=0,0 
for i in range(0,len(new)):
    if new[i] == test_y[i]:
        pos = pos+1
    else:
        neg = neg+1

print("\n\n ********** RESULTS BY LR ********** \n")

print("correctly classified     --> ",pos)
print("not correctly classified -->",neg)
print("percentage(%) accuracy   --> ",(float(pos)/float(len(new)))*100)
'''