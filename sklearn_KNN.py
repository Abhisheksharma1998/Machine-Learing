# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:20:48 2018

@author: Abhishek
"""

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
import pandas as pd

train = pd.read_csv("iris_training1.csv")
test = pd.read_csv("test.csv")

X_train = train.iloc[:,:-1]
X_test = test.iloc[:,:-1]
Y_train = train.iloc[:,4]
Y_test = test.iloc[:,4]

n=int(input("Enter the value of K: "))

c_obj=KNN(n_neighbors = n)
c_obj.fit(X_train,Y_train)
predict=c_obj.predict(X_test)
print("Accuray :",accuracy_score(predict,Y_test)*100)
