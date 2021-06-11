import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time

# Defining the two functions
def expo(x):
    return np.exp(x)

def cc(x):
    return np.cos(np.cos(x))
    
x_vals = np.arange(-2*np.pi,4*np.pi,0.01)
expo_x = expo(x_vals)
cc_x = cc(x_vals)

# Printing the expo(x) function as a semilog plot
plt.semilogy(x_vals,expo_x,'r')
plt.grid(True)
plt.ylabel(r'$e^{x}\rightarrow$',fontsize=13)
plt.xlabel(r'x$\rightarrow$',fontsize=13)
plt.title(r'Semi-Log Plot of $e^{x}$',fontsize=16)
plt.savefig("Figure1.png",dpi=1000)
plt.show()

# Printing the cos(cos(x)) function
plt.plot(x_vals,cc_x,'r')
plt.grid(True)
plt.ylabel(r'$\cos(\cos(x))\rightarrow$',fontsize=13)
plt.xlabel(r'x$\rightarrow$',fontsize=13)
plt.title(r'Plot of $\cos(\cos(x))$',fontsize=16)
plt.savefig("Figure2.png",dpi=1000)
plt.show()

# Fourier Series Coefficients via Integration

def cfnts_fourier(n,func):
    cfnts = np.zeros(n)
    f = func
    u = lambda x,k: f(x)*np.cos(k*x)
    v = lambda x,k: f(x)*np.sin(k*x)
    cfnts[0]= quad(f,0,2*np.pi)[0]/(2*np.pi)
    for i in range(1,n,2):
        cfnts[i] = quad(u,0,2*np.pi,args=((i+1)/2))[0]/np.pi
    for i in range(2,n,2):
        cfnts[i] = quad(v,0,2*np.pi,args=(i/2))[0]/np.pi
    return cfnts
    
# exp(x) fourier coefficients plots

t0 = time.time()
expo_cfnts = cfnts_fourier(51,expo)
t1 = time.time()
delTime1 = t1 - t0

plt.semilogy(range(51),abs(expo_cfnts),'ro')
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'Coefficient Magnitudes$\rightarrow$',fontsize=13)
plt.title('Semi-Log Plot of Fourier Series coefficients for $e^{x}$',fontsize=16)
plt.savefig("Figure3.png",dpi=1000)
plt.show()

plt.loglog(range(51),abs(expo_cfnts),'ro')
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'Coefficient Magnitudes$\rightarrow$',fontsize=13)
plt.title('Log-Log Plot of Fourier Series coefficients for $e^{x}$',fontsize=16)
plt.savefig("Figure4.png",dpi=1000)
plt.show()

# cos(cos(x)) fourier coefficients plots
t0 = time.time()
cc_cfnts = cfnts_fourier(51,cc)
t1 = time.time()
delTime2 = t1 - t0

plt.semilogy(range(51),abs(cc_cfnts),'ro')
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'Coefficient Magnitudes$\rightarrow$',fontsize=13)
plt.title('Semi-Log Plot of Fourier Series coefficients for $\cos(\cos(x))$',fontsize=16)
plt.savefig("Figure5.png",dpi=1000)
plt.show()

plt.loglog(range(51),abs(cc_cfnts),'ro')
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'Coefficient Magnitudes$\rightarrow$',fontsize=13)
plt.title('Log-Log Plot of Fourier Series coefficients for $\cos(\cos(x))$',fontsize=16)
plt.savefig("Figure6.png",dpi=1000)
plt.show()

# Coefficients via Least Squares

x = np.linspace(0,2*np.pi,401)
x = x[:-1] # drop last term to have a proper periodic integral
A = np.zeros((400,51)) # an empty matrix A to fill
A[:,0] = 1 # col 1 is all ones
for k in range(1,26):
    A[:,2*k-1] = np.cos(k*x) # cos(kx) column
    A[:,2*k] = np.sin(k*x) # sin(kx) column

def cfnts_lstsq(func,A,x):
    b = func(x) # func takes a vector input and returns a vector output
    cfnts=np.linalg.lstsq(A,b,rcond=None)[0]
    # the ’[0]’ is to pull out the best fit vector. lstsq returns a list.
    return cfnts

# exp(x) lstsq coefficients plots
t0 = time.time()
expo_lstsq = cfnts_lstsq(expo,A,x)
t1 = time.time()
delTime3 = t1 - t0

plt.semilogy(range(51),abs(expo_lstsq),'go')
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'Coefficient Magnitudes$\rightarrow$',fontsize=13)
plt.title('Semi-Log Plot of Fourier Series coefficients for $e^{x}$',fontsize=16)
plt.savefig("Figure7.png",dpi=1000)
plt.show()

plt.loglog(range(51),abs(expo_lstsq),'go')
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'Coefficient Magnitudes$\rightarrow$',fontsize=13)
plt.title('Log-Log Plot of Fourier Series coefficients for $e^{x}$',fontsize=16)
plt.savefig("Figure8.png",dpi=1000)
plt.show()

# cos(cos(x)) lstsq coefficients plots
t0 = time.time()
cc_lstsq = cfnts_lstsq(cc,A,x)
t1 = time.time()
delTime4 = t1 - t0

plt.semilogy(range(51),abs(cc_lstsq),'go')
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'Coefficient Magnitudes$\rightarrow$',fontsize=13)
plt.title('Semi-Log Plot of Fourier Series coefficients for $\cos(\cos(x))$',fontsize=16)
plt.savefig("Figure9.png",dpi=1000)
plt.show()

plt.loglog(range(51),abs(cc_lstsq),'go')
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'Coefficient Magnitudes$\rightarrow$',fontsize=13)
plt.title('Log-Log Plot of Fourier Series coefficients for $\cos(\cos(x))$',fontsize=16)
plt.savefig("Figure10.png",dpi=1000)
plt.show()

# Now, to compare them:

# # Comparing exp(x) coefficients

plt.semilogy(range(51),abs(expo_lstsq),'go',label='Least Squares Approach')
plt.semilogy(range(51),abs(expo_cfnts),'ro',label='Integration Approach')
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'Coefficient Magnitudes$\rightarrow$',fontsize=13)
plt.title('Semi-Log Plot of Fourier Series coefficients for $e^{x}$',fontsize=16)
plt.legend(loc='upper right')
plt.savefig("Figure11.png",dpi=1000)
plt.show()

plt.loglog(range(51),abs(expo_lstsq),'go',label='Least Squares Approach')
plt.loglog(range(51),abs(expo_cfnts),'ro',label='Integration Approach')
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'Coefficient Magnitudes$\rightarrow$',fontsize=13)
plt.title('Log-Log Plot of Fourier Series coefficients for $e^{x}$',fontsize=16)
plt.legend(loc='upper right')
plt.savefig("Figure12.png",dpi=1000)
plt.show()

# Comparing cos(cos(x)) coefficients

plt.semilogy(range(51),abs(cc_lstsq),'go',label='Least Squares Approach')
plt.semilogy(range(51),abs(cc_cfnts),'ro',label='Integration Approach')
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'Coefficient Magnitudes$\rightarrow$',fontsize=13)
plt.title('Semi-Log Plot of Fourier Series coefficients for $\cos(\cos(x))$',fontsize=16)
plt.savefig("Figure13.png",dpi=1000)
plt.show()

plt.loglog(range(51),abs(cc_lstsq),'go',label='Least Squares Approach')
plt.loglog(range(51),abs(cc_cfnts),'ro',label='Integration Approach')
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'Coefficient Magnitudes$\rightarrow$',fontsize=13)
plt.title('Log-Log Plot of Fourier Series coefficients for $\cos(\cos(x))$',fontsize=16)
plt.savefig("Figure14.png",dpi=1000)
plt.show()


# Finding the Differences and the Max Differences:

dev_expo = abs(expo_lstsq-expo_cfnts)
dev_cc = abs(cc_lstsq-cc_cfnts)

maxdev_expo = np.max(dev_expo)
maxdev_cc = np.max(dev_cc)

print(maxdev_expo)
print(maxdev_cc)

# Comparing times taken
print(delTime1)
print(delTime2)
print(delTime3)
print(delTime4)

# Function obtained from approximation:

# exp(x) Approximation vs Actual values:

expo_approx = A@expo_lstsq

plt.semilogy(x,expo_approx,'go',label='Approximation')
plt.semilogy(x,expo(x),'-r',label='Actual value')
plt.grid(True)
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'$f(x)\rightarrow$',fontsize=13)
plt.title('Plot of $e^{x}$ vs Fourier series approximation to 51 terms',fontsize=16)
plt.legend(loc='upper left')
plt.savefig('Figure15.png',dpi=1000)
plt.show()

cc_approx = A@cc_lstsq
plt.plot(x,cc_approx,'go',label='Approximation')
plt.plot(x,cc(x),'-r',label='Actual value')
plt.grid(True)
plt.xlabel(r'n$\rightarrow$',fontsize=13)
plt.ylabel(r'$f(x)\rightarrow$',fontsize=13)
plt.title('Plot of $\cos(\cos(x))$ vs Fourier series approximation to 51 terms',fontsize=16)
plt.legend(loc='upper left')
plt.savefig('Figure16.png',dpi=1000)
plt.show()
