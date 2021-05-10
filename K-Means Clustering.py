# -*- coding: utf-8 -*-
"""PR_assm5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RSontjvVSloUjwe1E-6GC2rbzraUzRRX
"""

#problem 1: plot all the points 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_dataset = np.loadtxt('/content/drive/MyDrive/4.2/pattern /lab5/data_k_mean.txt')
print(len(train_dataset))

x,y=train_dataset[:,0],train_dataset[:,1] # all values o column 0
#y=train_dataset[:,1]# all values o column 1

fig, ax = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(10)
ax.scatter(x,y,marker='o',color='r',label='Data Points')
legend = ax.legend(loc='upper left', shadow=False,labelspacing=0.5)
legend.get_frame().set_facecolor('None')
plt.show()
plt.savefig('TrainClass.png')

from math import sqrt
def euclidean_distance(x,y):
  euc_dis=sqrt(pow((x[0]-y[0]),2)+pow((x[1]-y[1]),2))
  return euc_dis

import random
random.seed(1)

random.shuffle(train_dataset)
print(train_dataset)

print(train_dataset[0][0],train_dataset[0][1])

def calculate_centroid(cluster):
    sum_x=0
    sum_y=0
    for i in cluster:
        sum_x+=train_dataset[i][0]
        sum_y+=train_dataset[i][1] 
    n=max(1,len(cluster))      
    mean_x=(sum_x)/n
    mean_y=(sum_y)/n
    return mean_x,mean_y

def update_cluster(means,k):
    update=[]
    for i in range(k):
        x=[]
        update.append(x)
    for i in range(len(train_dataset)):
        compare=9999  #distance big value
        clus=-1#new cluster
        for j in range(k):
            d=euclidean_distance(means[j],train_dataset[i])
            if(d<compare):
              clus=j
              compare=d 
        update[clus].append(i)
    return update 

def check(a,b,k):
    for i in range(k):
        if(len(a[i])!=len(b[i])):          
            return 0
        a[i].sort()
        b[i].sort()
        for j in range(len(a[i])):
            if(a[i][j]!=b[i][j]):
                return 0
    return 1                               
def k_means_clustering(k):
    val=[]
    for i in range(k):
        x=[]
        val.append(x)
    for i in range(k):
        val[i].append(i)   
    while(1):
        centroid=[]
        for t in range(k):
            centroid.append(calculate_centroid(val[t]))
        current_cluster=update_cluster(centroid,k)
        if(check(val,current_cluster,k)):
            break
        val=current_cluster.copy()
    return current_cluster

k=int(input('Enter a value for k:'))
final_cluster=k_means_clustering(k)  
print("The final cluster")  
for i in range(k):
    print(final_cluster[i])
x_train=[]
y_train=[]
for i in range(k):#size of the array
    x,y=[],[]
    x_train.append(x)
    y_train.append(y)
for i in range(k):
    for j in final_cluster[i]:
      x_train[i].append(train_dataset[j][0])
      y_train[i].append(train_dataset[j][1])

fig,ax=plt.subplots()
colors=['red','blue','green','yellow','orange','c','m','y','k']
marker=['o','v','1','^','s','p','+','D','X','P','2','H','3','d','4']
for i in range(k):
    name="Cluster "+str(i+1)
    ax.scatter(x_train[i],y_train[i],marker=marker[i],color=colors[i],label=name)
fig.set_figheight(8)
fig.set_figwidth(10)
legend.get_frame().set_facecolor('None')
ax.legend()
plt.show()
plt.savefig('4.png')