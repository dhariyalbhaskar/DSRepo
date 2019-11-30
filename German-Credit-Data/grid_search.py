#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 23:03:39 2016

@author: bhaskar
"""

import csv 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


# Loading the Digits dataset
csv_file_object=csv.reader(open('german.csv'))
data=[]
for row in csv_file_object:
    data.append(row)
    
data=np.array(data, dtype=np.float64)
new_data=data[:,:-1] 
# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
X = new_data[:,:-1]
y = new_data[:,-1]

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,1e-5,1e-6,1e-7],
                     'C': [1, 10, 100, 1000]}
                    ]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10,
                       scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))