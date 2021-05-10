
'''Problem 1 '''
import numpy as np
train_dataset = np.loadtxt('/content/drive/MyDrive/4.2/pattern /test-Minimum-Error-Rate-Classifier.txt',delimiter=",",dtype='float64')
print(train_dataset)
#Given,
w1=w2=0.5
cov1=np.array([[.25,.3],[.3,1]])
cov2=np.array([[.5,.0],[0,.5]])
mean1=np.array([0,0])
mean2=np.array([2,2])
print(mean1,w1,w2)
print( cov1, cov2)

D = mean1.shape[0]
def normal_distribution(mean, cov, x):  
    cov_det = np.linalg.det(cov) #finding the determinant of covarience 
    cov_inv = np.linalg.inv(cov) #finding the inverse of covarience 
    N = np.sqrt((2*np.pi)**D*cov_det) 
    xu_T=  np.transpose(x - mean) #(x-mu)T    
    xu = (x- mean) #(x-mu)
    mul  = 0.5 * (np.dot(xu_T,cov_inv).dot(xu))           
    expo= np.exp(-mul / 2) 
    result= expo/ N
    return result

x1=[]
y1=[]
x2=[]
y2=[]
for train in  train_dataset:     
    post1=w1*normal_distribution(mean1,cov1,np.array([train[0],train[1]])) 
    post2=w2*normal_distribution(mean2,cov2,np.array([train[0],train[1]]))
    print( post1, post2)
    if( post1< post2):
        x2.append(train[0])
        y2.append(train[1])
    else:
        x1.append(train[0])
        y1.append(train[1])
print("Class1",x1,y1)
print("Class2",x2,y2)
import pandas as pd
import matplotlib.pyplot as plt
fig=plt.figure(1, figsize=(10,7))
plt.title("Problem 1 output")
plt.xlabel('x-axis', color='black')
plt.ylabel('y-axis', color='black')
chart1= fig.add_subplot()
chart1.scatter(x1,y1,marker='o',color='r',label='Train class 1');
chart1.scatter(x2,y2,marker='*',color='k',label='Train class 2');

chart1.axis([-5,10,-5,20]);
chart1.legend()#
#plt.savefig('TrainClass.png')

D = mean1.shape[0]
def gaussian_distribution1(mean, cov, x):
  cov_det = np.linalg.det(cov) #finding the determinant of covarience 
  cov_inv = np.linalg.inv(cov) #finding the inverse of covarience 
  N = np.sqrt((2*np.pi)**D*cov_det)  # 2*pi^D*determ
  mul = np.einsum('...i,ij,...j->...', x-mean,cov_inv, x-mean) #explicit mode summation: desired summation 
  expo= np.exp(-mul / 2) 
  result= expo/ N
  return result

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
X = np.linspace(-8, 8, 40)
Y = np.linspace(-8,8 , 40)
X, Y = np.meshgrid(X, Y)
x= np.empty(X.shape + (2,))
x[:, :, 0] = X
x[:, :, 1] = Y
Z = gaussian_distribution1( mean1, cov1,x)
Z1 = gaussian_distribution1( mean2, cov2,x)
fig = plt.figure()
ax = fig.gca(projection='3d')
fig.set_figheight(6)
fig.set_figwidth(8)
z=0
ax.scatter(x1,y1,color='red',alpha=1)
ax.scatter(x2,y2,color='blue',alpha=1)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,linewidth=1, antialiased=True,cmap=cm.Oranges,alpha=.5)
ax.plot_surface(X, Y, Z1, rstride=1, cstride=1,linewidth=1, antialiased=True,cmap=cm.ocean,alpha=.5)
ax.set_title('Probability Density')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability Density')
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(.30,0.0,6))
ax.view_init(27, -102)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
X = np.linspace(-8, 8, 40)
Y = np.linspace(-8,8 , 40)
X, Y = np.meshgrid(X, Y)
x= np.empty(X.shape + (2,))
x[:, :, 0] = X
x[:, :, 1] = Y
Z = gaussian_distribution1( mean1, cov1,x)
Z1 = gaussian_distribution1( mean2, cov2,x)
db=(Z-Z1)
fig = plt.figure()
ax = fig.gca(projection='3d')
fig.set_figheight(6)
fig.set_figwidth(8)
z=0
ax.scatter(x1,y1,color='red',alpha=1)
ax.scatter(x2,y2,color='blue',alpha=1)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,linewidth=1, antialiased=True,cmap=cm.Oranges,alpha=.5)
ax.plot_surface(X, Y, Z1, rstride=1, cstride=1,linewidth=1, antialiased=True,cmap=cm.ocean,alpha=.5)
ax.contourf(X, Y, db, zdir='z', offset=-.15,cmap=cm.cool,alpha=0.5)
ax.set_title('Probability Density')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability Density')
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(.30,0.0,6))
ax.view_init(27, -102)
plt.show()