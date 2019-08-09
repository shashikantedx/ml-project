# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 19:36:49 2019

@author: shashikant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


dataset=pd.read_csv('dataset_kaggle.csv')
dataset=dataset.drop(['index'],axis=1)

list=[0,1,6,10,11,12,13,14,15,16,18,21,23,25,27]

X=dataset.iloc[:,list]
y=dataset.loc[:,['Result']]  #left side for row and right side for column




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)



sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, pred_rfc))










