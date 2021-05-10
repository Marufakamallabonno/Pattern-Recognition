# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:08:57 2021

@author: Labonno
"""
import matplotlib.pyplot as plt
import numpy as np
train_data =np.int32(np.loadtxt('train.txt'));
test_data =np.int32(np.loadtxt('test.txt'));
#print(train_data);
#print(test_data);
print(train_data[1][1]);
train_data_c1=[];
train_data_c2=[];
for item in train_data:
    if item[2]==1:
        #print(item[2]);
        train_data_c1.append(item)
    elif item[2]==2:
        train_data_c2.append(item)
train_data_class1= np.array(train_data_c1);
train_data_class2= np.array(train_data_c2);
#
print(train_data_class1);
print(train_data_class2);
x1,y1= train_data_class1[:,0],train_data_class1[:,1];
x2,y2= train_data_class2[:,0],train_data_class2[:,1];
print("x1",x1);
print("x2",x2);
fig=plt.figure(1, figsize=(10,10))
chart1= fig.add_subplot()
chart1.scatter(x1,y1,marker='o',color='r',label='Train class 1');
chart1.scatter(x2,y2,marker='*',color='k',label='Train class 2');

chart1.axis([-5,10,-5,20]);
chart1.legend()#
