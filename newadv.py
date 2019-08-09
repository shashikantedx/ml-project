# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:04:50 2019

@author: shashikant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


#importing the dataset
dataset = pd.read_csv("dataset_kaggle.csv")
#dataset = dataset.drop('id', 1) #removing unwanted column
dataset=dataset.drop(['index'],axis=1)

list=[0,1,6,10,11,12,13,14,15,16,18,21,23,25,27]

x=dataset.iloc[:,0:30]
y=dataset.loc[:,['Result']]


#spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =0 )

#fitting RandomForest regression with best params
classifier = RandomForestClassifier(n_estimators = 50, criterion = "gini", max_features = 'log2',  random_state = 0)
classifier.fit(x_train, y_train)

#predicting the tests set result
y_pred = classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#pickle file joblib
joblib.dump(classifier, 'rf_final.pkl')


#-------------Features Importance random forest
importances =classifier.feature_importances_
names = dataset.iloc[:,0:30].columns
sorted_importances = sorted(importances, reverse=True)
indices = np.argsort(-importances)
var_imp = pd.DataFrame(sorted_importances, names[indices], columns=['importance'])



#-------------plotting variable importance
plt.title("Variable Importances")
plt.barh(np.arange(len(names)), sorted_importances, height = 0.7)
plt.yticks(np.arange(len(names)), names[indices], fontsize=7)
plt.xlabel('Relative Importance')
plt.show()
