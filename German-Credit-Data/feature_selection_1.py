# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 03:55:08 2016

@author: ajay
"""

#My code on german data
# Recursive Feature Elimination

import csv 
import numpy as np
from sklearn import preprocessing, svm, linear_model, metrics,cross_validation,neighbors
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import scipy

# load the datasets
csv_file_object=csv.reader(open('german.csv'))
data=[]
for row in csv_file_object:
    data.append(row)
data=np.array(data, dtype=np.float64)
dataset=data[:,:-1]

target=data[:,-1]
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

#==========================================================================
new_data=dataset

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

#NN--------------------------------------------------------------------------------
X=normalized_data
y=data[:,-1]

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

clf=MLPClassifier(solver='lbfgs',activation='logistic',alpha=1,hidden_layer_sizes=(5,5),learning_rate_init=0.75,random_state=0)
clf.fit(X_train,y_train)


y_pred=clf.predict(X_test)

e_ar=e_ar+ [y_pred]
print("\n\n ********** RESULTS BY MLP ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

#PCA+SVM-------------------------------------------------------

pca=PCA(n_components=15)
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

#PCA+LR-------------------------------------------------------------

pca=PCA(n_components=15)
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


#NN--------------------------------------------------------------------------------
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

