# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:50:24 2017

@author: Abhishek
"""
import pandas as pd
from sklearn import svm
df = pd.read_csv("dataset_Facebook.csv")

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1,2))
x_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(x_scaled)


import numpy as np
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    

from sklearn.model_selection import KFold # import KFold
X=df.iloc[:,:7].values
#Y=df.iloc[:,7:18].values
kf = KFold(n_splits=10,shuffle=False) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf) 

for i in range(7,19):
    final_sum=0
    y=df.iloc[:, i].values
    for train_index, test_index in kf.split(X):
        #print 'TRAIN:', train_index
        #print 'TEST:', test_index
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = svm.SVR(C=1000, cache_size=500, coef0=0.0, degree=0, epsilon=0.1, gamma='auto',
                        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
        model.fit(X_train, y_train)
        model.score(X_train, y_train)
        predicted= model.predict(X_test)
        result = mean_absolute_percentage_error(y_test,predicted)
        final_sum = final_sum + result
    final_sum=final_sum/10
    print final_sum
              