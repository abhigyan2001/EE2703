import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

def f(t,A,B):
    return A*sp.jn(2,t) + B*t

data_columns = []
data_columns = np.loadtxt('fitting.dat',dtype=float)
time  = np.array(data_columns[:,0])
y_columns = np.asarray(data_columns)[:,1:]
sigma = np.logspace(-1,-3,9)

# plotting all the data from the file:

plt.figure(figsize=(7,6))
for i in range(9):
    plt.plot(time,y_columns[:,i],label=r'$\sigma$=%.3f'%sigma[i])
plt.plot(time,f(time,1.05,-0.105),'-k',label='True curve')
plt.legend(loc='upper right')
plt.ylabel(r'f(t)+noise$\rightarrow$',fontsize=15)
plt.xlabel(r't$\rightarrow$',fontsize=15)
plt.title("Q4: Data to be fitted to theory")
plt.savefig("Q4.png",dpi=1000)
plt.show()


data = y_columns[:,0]
plt.errorbar(time[::5],data[::5],sigma[0],fmt='ro',label='Errorbar')
plt.plot(time,f(time,1.05,-0.105),'b',label='$f(t)$')
plt.legend(loc='upper right')
plt.xlabel(r't$\rightarrow$',fontsize=15)
plt.title("Q5: Data points for $\sigma=0.10$ along with exact function")
plt.savefig("Q5.png",dpi=1000)
plt.show()



fn_column = sp.jn(2,time)
M = np.c_[fn_column,time]


A = 1.05; B = -0.105
C = np.array([A,B]) # the parameter array
G = M @ C # matrix obtained by matrix multiplication


G1 = np.array(f(time,A,B)) # matrix obained direcrly from the function

# For testing equality
# print(np.array_equal(G,G1)) # uncomment this line to test it


#initialization
e = np.zeros((21,21,9))
A = np.linspace(0,2,21)

B = np.linspace(-0.2,0,21)

for k in range(9):
    f1 = y_columns[:,k]
    for i in range(21):
        for j in range(21):
            e[i][j][k] = np.sum((f1 -np.array(f(time,A[i],B[j])))**2)/101

# plotting the contour and locating the minima

plot = plt.contour(A,B,e[:,:,0],20)
plt.ylabel(r'B$\rightarrow$')
plt.xlabel(r'A$\rightarrow$')
plt.clabel(plot,inline=1,fontsize=10)

# Using np.unravel_index to obtain the location of the minima in the original array

a = np.unravel_index(np.argmin(e[:,:,0]),e[:,:,0].shape)
plt.plot(A[a[0]],B[a[1]],'o',markersize=3)
plt.annotate('(%0.2f,%0.2f)'%(A[a[0]],B[a[1]]),(A[a[0]],B[a[1]]))
plt.title("Q8: contour plot of $\epsilon_{ij}$")
plt.savefig("Q8.png",dpi=1000)
plt.show()

est = [np.linalg.lstsq(M,y_columns[:,i])[0] for i in range(9)]
est = np.asarray(est)

# Obtaining the error in A and B 

error_a = abs(est[:,0]-1.05)
error_b = abs(est[:,1]+0.105)

plt.plot(sigma,error_a,'ro--',label='Aerr',linewidth=0.25,dashes=(20,50))
plt.plot(sigma,error_b,'go--',label='Berr',linewidth=0.25,dashes=(20,50))
plt.ylabel(r'MS Error$\rightarrow$',fontsize=15)
plt.xlabel(r'Noise standard deviation $(\sigma_{n})\rightarrow$',fontsize=15)
plt.legend(loc='upper left')
plt.title("Q10: Variation of error with noise")
plt.savefig("Q10.png",dpi=1000)
plt.show()

plt.figure(figsize=(6,6))
plt.loglog(sigma,error_a,'ro')
plt.stem(sigma,error_a,'-ro')
plt.loglog(sigma,error_b,'go')
plt.stem(sigma,(error_b),'-go')
plt.xlabel(r'$\sigma_{n}\rightarrow$',fontsize=15)
plt.ylabel(r'MS Error$\rightarrow$',fontsize=15)
plt.title("Q11: Variation of Error with noise")
plt.savefig("Q11.png",dpi=1000)
plt.show()