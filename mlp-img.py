import cv2
import numpy as np

img_size=50
m=50
classes=3
inlayer=np.zeros((m*classes,img_size*img_size))
letters=['a','b','c']
k=0
for j in letters:
	print(j)
	for i in range(1,m+1):
		print(i)
		if(i<10):
			path='/home/agetic/dev/python/machine_learning/letras/'+j+'00'+str(i)+'.JPG'
		else:
			path='/home/agetic/dev/python/machine_learning/letras/'+j+'0'+str(i)+'.JPG'
		print(path)
		img=cv2.imread(path)
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		resized_img=cv2.resize(gray,(img_size,img_size))
		inlayer[k*m+i-1,:]=resized_img.flatten()[np.newaxis]
		cv2.imshow('RESIZED',resized_img)
		cv2.waitKey(10)
	k=k+1
inlayer=inlayer.astype(float)/255
outlayer=np.zeros((inlayer.shape[0],classes))
outlayer[0:m,:]=np.array([1,0,0])
outlayer[m:2*m,:]=np.array([0,1,0])
outlayer[2*m:3*m,:]=np.array([0,0,1])


(m,n)=inlayer.shape
hidden_layer_size=10
output_layer_size=outlayer.shape[1]
X=np.ones((m,n+1))
X[:,1:n+1]=inlayer

weight_1=(np.random.rand(hidden_layer_size,n+1)*4.8-2.4)/(n+1)
weight_2=(np.random.rand(output_layer_size,hidden_layer_size+1)*4.8-2.4)/(hidden_layer_size+1)
alpha=0.1
epochs=5000
for j in range(0,epochs):
	print(j)
	for i in range(0,m):
		a_1=X[i,:]
		Z_1=1/(1+np.exp(-(np.dot(a_1,weight_1.T))))
		a_2=np.ones((Z_1.shape[0]+1))
		a_2[1:hidden_layer_size+1]=Z_1
		Z_2=1/(1+np.exp(-(np.dot(a_2,weight_2.T))))

		e=outlayer[i,:]-Z_2
		grad_k=Z_2*(1-Z_2)*e
		delta_2=alpha*np.dot(grad_k[np.newaxis].T,a_2[np.newaxis])
		grad_j=Z_1*(1-Z_1)*(np.dot(grad_k[np.newaxis],weight_2[:,1:]))
		delta_1=alpha*np.dot(grad_j.T,a_1[np.newaxis])
		weight_1=weight_1+delta_1
		weight_2=weight_2+delta_2


pred=np.zeros(outlayer.shape)
for i in range(0,m):
	a_1=X[i,:]
	Z_1=1/(1+np.exp(-(np.dot(a_1,weight_1.T))))
	a_2=np.ones((Z_1.shape[0]+1))
	a_2[1:hidden_layer_size+1]=Z_1
	pred[i]=1/(1+np.exp(-(np.dot(a_2,weight_2.T))))
print(pred)

np.savetxt("w1.csv",weight_1 , delimiter=",")
np.savetxt("w2.csv",weight_2 , delimiter=",")
cv2.destroyAllWindows()

nimg=cv2.imread('/home/agetic/dev/python/machine_learning/letras/a003.JPG')
gray=cv2.cvtColor(nimg,cv2.COLOR_BGR2GRAY)
resized_img=cv2.resize(gray,(img_size,img_size))
nletra=resized_img.flatten()[np.newaxis]
print(nletra)
