import numpy as np
import matplotlib.pyplot as plt

X=np.array([[0.5, 0.5],[1, 0.5],[2, 0.3],[0.7, 1],
            [1.5, 0.8],[0.4,1.5],[1,1.5],[0.5,2.5],
            [3,1.1],[2.8,1.8],[4,2.8],[2.5,2.3],
            [3,2.3],[2.5,3],[3,3],[1.5,3.5],
            [2.3,3.6],[2.8,3.5],[4,3.2],[2,4.5]])

neurons = 8
(m,n) = X.shape #m: training examples n: features
w=np.random.rand(neurons,n)
print(w)
iterations=1000
alpha=0.06
D=np.zeros(neurons)

fig2=plt.figure()
ax2=fig2.add_subplot(1,1,1,facecolor="1.0")
ax2.plot(w[:,0],w[:,1],'ro')
plt.show()

for i in range(0, iterations):
    if (i==200):
        fig4=plt.figure()
        ax4=fig4.add_subplot(1,1,1,facecolor="1.0")
        ax4.plot(w[:,0],w[:,1],'go')
    for j in range(0, m):
        for k in range(0, neurons):
            D[k]=np.sqrt(np.sum(np.square(X[j,:]-w[k,:])))
        win=(D==np.min(D)) # win es un vector de 1s y 0s para el ganador
        for k in range(0, neurons):
            dw=win[k]*alpha*(X[j,:]-w[k,:])
            print(w)
            w[k,:]=w[k,:]+dw
        print("----------")
fig1=plt.figure()
ax1=fig1.add_subplot(1,1,1,facecolor="1.0")
ax1.plot(X[:,0], X[:,1],'ro')
plt.show()

fig3=plt.figure()
ax3=fig3.add_subplot(1,1,1,facecolor="1.0")
ax3.plot(w[:,0],w[:,1],'bo')
plt.show()