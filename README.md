# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Vishwa vasu. R
RegisterNumber:212222040183
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]

x[:5]

y[:5]

plt.figure()
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted",color="cadetblue")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted",color="plum")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot),color="cadetblue")
plt.show()

def costFunction(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j= -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h)))/x.shape[0]
  return j


def gradient(theta,x,y):

  h=sigmoid(np.dot(x,theta))
  grad=np.dot(x.T,h-y)/x.shape[0]
  return grad


x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,x,y):
  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted",color="mediumpurple")
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted",color="pink")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,x,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
  prob=sigmoid(np.dot(x_train,theta))
  return(prob>=0.5).astype(int)

np.mean(predict(res.x,x)==y)
 
*/
```

## Output:
![ex 5 1](https://github.com/vishwa2005vasu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135954202/ef72bbed-ee12-4a38-8954-e9b666b5d046)
![ex 5 2](https://github.com/vishwa2005vasu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135954202/7e30a4b7-29dd-4fe1-b6e7-dfa7dd66bc79)
![ex 5 3](https://github.com/vishwa2005vasu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135954202/2c1ff569-d9fb-4b36-80fe-954fde57f433)
![ex 5 4](https://github.com/vishwa2005vasu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135954202/5e5746c3-c5c6-4ef1-bd97-7d85c73019f9)
![ex 5 5](https://github.com/vishwa2005vasu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135954202/92c1e0c7-cc5c-47a4-90e2-382f1ea364c6)
![ex 5 6](https://github.com/vishwa2005vasu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135954202/adfea1a6-96d8-4d68-aab3-b82edc5924ed)
![ex 5 7](https://github.com/vishwa2005vasu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135954202/f37fec28-c80f-4f95-8c8c-db583cc44a34)
![ex 57](https://github.com/vishwa2005vasu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135954202/3d8d3fe0-e510-4ce8-a118-b3b8886eb957)
![ex 5 8](https://github.com/vishwa2005vasu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135954202/fa1b6383-a18a-4119-b250-df5506be40de)
![ex 5 9](https://github.com/vishwa2005vasu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135954202/0638fc04-c54e-47c3-8ea0-2e98d0b7f6b5)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

