# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph. 


## Program:
```
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

Developed By:L.Mahesh Muthu

```

## Output:




 1.Array Value of x



 
![Screenshot (59)](https://github.com/MaheshMuthuL/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135570619/914d94f2-effb-4172-bf91-d3881be0129c)





 2.Array Value of y





 
![Screenshot (60)](https://github.com/MaheshMuthuL/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135570619/4c3a74a6-05e4-4c29-82bf-b3f5fd515c01)





 3.Exam 1-Score Graph





 
![Screenshot (61)](https://github.com/MaheshMuthuL/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135570619/0d784217-b371-43c9-8359-05956190c002)








 4.Sigmoid function graph




 
![Screenshot (62)](https://github.com/MaheshMuthuL/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135570619/dcf2473b-962f-44fb-bf1c-4541230186de)






 5.x_train_grad value





 
![Screenshot (63)](https://github.com/MaheshMuthuL/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135570619/a72cadd7-50ca-4355-8dcd-85d74038fa49)







 6.y_train_grad value






 
![Screenshot (64)](https://github.com/MaheshMuthuL/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135570619/86158786-18da-4d81-8b8e-12f859b7a116)







7.Print res.x




![Screenshot (65)](https://github.com/MaheshMuthuL/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135570619/00f12e94-2fd7-49e2-bfb7-d121521f4c3e)




8.Decision boundary-graph for exam score





![Screenshot (66)](https://github.com/MaheshMuthuL/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135570619/c5b2fe28-701b-44fc-a953-f1fa5e6723c5)





9.Probability value






![Screenshot (67)](https://github.com/MaheshMuthuL/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135570619/dd605d25-c387-4606-990f-88968dba59b7)






10.Prediction value of mean







![Screenshot (68)](https://github.com/MaheshMuthuL/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135570619/c78afc2e-f098-4e59-b95a-8dd0a776a3bc)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
