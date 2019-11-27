import numpy as np
import matplotlib.pyplot as plt

Xi=np.array([[-2,-3],[2,2],[3,1],[4,2],[-5,3],[-5,4],[-3,-4],[-4,-5],[-5,6],[-6,6],
             [4,2],[5,4],[6,9]])
print(Xi)
Y=np.array([0,1,1,1,2,2,0,0,2,2, 1,1,1])
print(Y)
m=Xi.shape[0]
n=Xi.shape[1] #features
classes=len(np.unique(Y))
aY=np.zeros((m,classes))

for i in range(0,classes):
    aY[np.where(Y==np.unique(Y)[i]),i]=1

print(aY)

X=np.ones((m,n+1))
print(X)
X[:,1:n+1]=Xi
theta=np.zeros((n+1,classes))
alpha=1
iterations=1000
J=np.zeros((iterations))

for i in range(0,iterations):
    theta=theta-alpha*np.dot(X.T, (1/(1+np.exp(-np.dot(X,theta)))-aY))/m
    hx=(1/(1+np.exp(-np.dot(X,theta))))
    J[i] =-np.sum(np.multiply(aY,np.log(hx)) + np.multiply(1-aY,np.log(1-hx)))/m

print(hx)
pred=np.zeros(Y.shape)
for i in range(0,m):
    pred[i]=np.unique(Y)[np.where(hx[i,:]==np.max(hx[i,:]))[0][0]]

print(pred)
