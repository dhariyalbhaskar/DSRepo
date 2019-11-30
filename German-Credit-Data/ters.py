#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 23:59:44 2016

@author: bhaskar
"""

import csv 
import numpy as np
csv_file_object=csv.reader(open('ge.csv'))

data=[]
for row in csv_file_object:
    data.append(row)
    
data=np.array(data, dtype=np.float64)