# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
train = pd.read_csv("iris_training.csv")
test = pd.read_csv("iris_test.csv")
trn=[[]]
tst=[[]]
for i in range(0,120):
    trn.append(train.iloc[i])

for i in range(0,30):
    tst.append(test.iloc[i])

final=[]
n=int(input("Enter the value of K :"))
import operator

import math as m
for i in range(1,31):
    dic=dict()
    for j in range(1,121):
        s=0
        for k in range(0,4):
            s=s+(trn[j][k]-tst[i][k])**2
        d = m.sqrt(s)
        dic[d]=trn[j][4]
    temp=sorted(dic.items(),key=operator.itemgetter(0))
    new=[]
    for it in range(0,n):
        new.append(temp[it][1])

    final.append(max(new,key=new.count))
count=0
for i in range(0,30):
    if(final[i] != tst[i+1][4]):
        count+=1
print("Accuray Percentage = ",((30-count)/30)*100)