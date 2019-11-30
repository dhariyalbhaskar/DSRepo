# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 18:05:15 2016

@author: bhaskar
"""
import csv as csv
import numpy as np

data = open('german.csv')

print(data)
x=[]
y=[]
for i in data.readlines():
    c=i.split('\n')
    d=c[0].split('"')
    e=d[1].split()
    b=[]
    for j in e[:-1]:
        b.append(float(j))
    #print b
    x.append(b)
    y.append(int(e[len(e)-1]))