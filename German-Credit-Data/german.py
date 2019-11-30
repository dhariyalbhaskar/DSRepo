# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 17:52:01 2016

@author: bhaskar
"""

import csv 
import numpy as np
from sklearn import preprocessing, svm, linear_model, metrics,cross_validation
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn import preprocessing as prep
#--------fetching data
csv_file_object=csv.reader(open('german.csv'))
#header=next(csv_file_object)


#--------------------------------
data=[]
for row in csv_file_object:
    data.append(row)
    
    
data=np.array(data, dtype=np.float64)
new_data=data[:,:-1]    #this is training data

#Data tranformation using MinMaxScaler
minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
normalized_data=minmax.fit_transform(new_data)      #normalizing training data
print('Nomalized Data')
#print(normalized_data)

#Principal component analysis
print('Principle component analysis') 
pca=PCA(n_components=20)
pca_X=pca.fit_transform(new_data)
#print(pca_X)

var1=np.cumsum(np.round(pca.explained_variance_, decimals=4)*100)
print('PCA Cum Var')
#print(var1)

#SVM  & PCA
X=pca_X
y=data[:,-1]

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)

model = svm.SVC(kernel='rbf', C=10, gamma=1e-3) 
model.fit(X_train,y_train)
y_pred=model.predict(X_test)



print("\n\n ********** RESULTS BY SVM+PCA ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)


print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

#SVM only
X=normalized_data
y=data[:,-1]
X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)

model = svm.SVC(kernel='rbf', C=10, gamma=0.001 )
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

        
print("\n\n ********** RESULTS BY SVM ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))


#Logistic regression
X=pca_X
y=data[:,-1]
X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)


logreg=linear_model.LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
#new = list(y_pred)


print("\n\n ********** RESULTS BY PCA+LR ********** \n")

print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))



train_x=new_data[:800,:-1]
train_x=np.array(train_x)
train_y=new_data[:800,-1]

test_x=new_data[800:,:-1]
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

#print("\n\n ********** RESULTS BY RF ********** \n")

X=normalized_data
y=data[:,-1]
X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.2,random_state=1)
model= RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
y_pred= model.predict(X_test)

print("\n\n ********** RESULTS BY RF ********** \n")
print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.8, max_depth=3)
clf.fit(X_train, y_train)

y_pred= model.predict(X_test)

print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

#Neural nets----------------------------------
'''
X=normalized_data
y=data[:,-1]

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)

clf=MLPClassifier(solver='lbfgs',alpha=1e-6,hidden_layer_sizes=(100,100),learning_rate_init=0.155,random_state=1)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("\n\n ********** RESULTS BY MLP ********** \n")

print('Accuracy=',metrics.accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix\n',metrics.confusion_matrix(y_test,y_pred))

#-----------------------------------------------------------------------

print("\nAddaboost LR")
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

X=normalized_data
y=data[:,-1]

#X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)
 
clf = AdaBoostClassifier(base_estimator=logreg,n_estimators=50,learning_rate=0.8)
scores = cross_val_score(clf, X, y)
print("ACC",scores.mean())
'''