# script to generate data files for the least squares assignment
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
N=101                           # no of data points
k=9                             # no of sets of data with varying noise

# generate the data points and add noise
t=np.linspace(0,10,N)           # t vector
y=1.05*sp.jn(2,t)-0.105*t       # f(t) vector
Y=np.meshgrid(y,np.ones(k),indexing='ij')[0] # make k copies
scl=np.logspace(-1,-3,k)       # noise stdev
n=np.random.randn(N,k)@np.diag(scl)   # generate k vectors
yy=Y+n                          # add noise to signal

# shadow plot
plt.plot(t,yy)
plt.xlabel(r'$t$',size=20)
plt.ylabel(r'$f(t)+n$',size=20)
plt.title(r'Plot of the data to be fitted')
plt.grid(True)
np.savetxt("fitting.dat",np.c_[t,yy]) # write out matrix to file
plt.show()
