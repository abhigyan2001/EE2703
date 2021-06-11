import numpy as np
from sympy import *
import scipy.signal as sp
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000

init_session
    
def sympy_to_lti(symp, s=symbols('s')):
    """Convert Sympy transfer function polynomial to Scipy TransferFunctionContinuous"""
    n, d = fraction(symp)  # expressions of numerator and denominator
    p_num_den = poly(n, s), poly(d, s)
    num, den = p_num_den[0].all_coeffs(), p_num_den[1].all_coeffs()
    return sp.lti(np.array(num,dtype=float), np.array(den,dtype=float))

# Setting up the Low Pass Butterworth Filter
s = symbols('s')
G = 1.586
R1 = R2 = 1e4
C1 = C2 = 1e-9
V1 = 1/s
A = Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0],[0,-G,G,1],[-1/R1 -1/R2 -s*C1, 1/R2, 0, s*C1]])
b = Matrix([0,0,0,-V1/R1])
V = A.inv()*b


# Plotting its Frequency Response
Vo = V[3]

H = sympy_to_lti(Vo)

print(H)
omega = np.logspace(0,8,801)
ss = 1j*omega
hf = lambdify(s, Vo, 'numpy')
v = hf(ss)
plt.figure(0)
plt.subplot(2,1,1)
plt.loglog(omega,abs(v))
plt.xlabel(r"$\omega \rightarrow$")
plt.ylabel(r"$|H(j \omega)|\rightarrow$")
plt.grid(True)

plt.subplot(2,1,2)
plt.semilogx(omega,np.angle(v))
plt.xlabel(r"$\omega \rightarrow$")
plt.ylabel(r"$\angle H(j \omega) \rightarrow$")
plt.grid(True)

plt.title("Magnitude Response of Butterworth Lowpass Filter")
plt.savefig("images/fig0",dpi=1000)
plt.show()


# Question 1: Step Response of given Butterworth Filter
t = np.linspace(0,0.001,1000)
t1,Vo1 = sp.step(H,T=t)

plt.figure(1)
plt.plot(t1,Vo1,"-",label=r"$V_0$")
plt.xlabel(r't$\rightarrow$')
plt.ylabel(r'$V_{o}\rightarrow$')
plt.title('Lowpass Butterworth Filter Step Response')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig("images/fig1",dpi=1000)
plt.show()

# Question 2: Response of Butterworth filter to mixed frequency sinusoid
t = np.linspace(0,0.01,100000)
Vi = np.multiply((np.sin(2000*np.pi*t)+np.cos(2000000*np.pi*t)),np.heaviside(t,0.5))
Vo = sp.lsim(H,Vi,T=t)
plt.figure(2)
plt.plot(Vo[0],Vi,label=r'$V_{in}$')
plt.plot(Vo[0],Vo[1],label=r'$V_{out}$')
plt.xlabel(r't$\rightarrow$')
plt.ylabel(r'V$\rightarrow$')
plt.title("Filter Response for Mixed Frequency Sinusoid")
plt.legend(loc="upper right")
plt.savefig("images/fig2", dpi=1000)
plt.show()

# Setting up the High Pass Butterworth Filter
G = 1.586
R1 = R3 = 1e4
C1 = C2 = 1e-9
Vi = 1
A=Matrix([[0,0,1,-1/G],[-1/(1+1/(s*R3*C2)),1,0,0],[0,-G,G,1],[s*-C1-s*C2-1/R1,s*C2,0,1/R1]])
b=Matrix([0,0,0,-Vi*s*C1])
V = A.inv()*b

# Question 3: Magnitude Response of the Filter
Vo = V[3]
H = sympy_to_lti(Vo)
omega = np.logspace(0,8,801)
ss = 1j*omega
hf = lambdify(s, Vo, 'numpy')
v = hf(ss)
plt.figure(3)
plt.loglog(omega,abs(v))
plt.xlabel(r"$\omega \rightarrow$")
plt.ylabel(r"$|H(j \omega)|\rightarrow$")
plt.grid(True)
plt.title("Magnitude Response of Butterworth Highpass Filter")
#plt.savefig("images/fig3",dpi=1000)
plt.show()

# Question 4: Highpass Filter response to Damped Sinusoid:
t = np.linspace(0,10,1000)
Vi = np.multiply(np.multiply(np.exp(-0.5*t),np.sin(2*np.pi*t)),np.heaviside(t,0.5))
Vo = sp.lsim(H,Vi,T=t)
plt.figure(4)
plt.plot(Vo[0],Vi,label=r'$V_{in}$')
plt.plot(Vo[0],Vo[1],label=r'$V_{out}$')
plt.xlabel(r't$\rightarrow$')
plt.ylabel(r'$V\rightarrow$')
plt.title(r"Response for decaying sinusoid of frequency $2$ Hz")
plt.savefig("images/fig4",dpi=1000)
plt.show()

t = np.linspace(0,0.0001,10000)
Vi = np.multiply(np.multiply(np.exp(-0.5*t),np.sin(2*np.pi*200000*t)),np.heaviside(t,0.5))
Vo = sp.lsim(H,Vi,T=t)
plt.figure(5)
plt.plot(Vo[0],Vi,label=r'$V_{in}$')
plt.plot(Vo[0],Vo[1],label=r'$V_{out}$')
plt.xlabel(r't$\rightarrow$')
plt.ylabel(r'$V\rightarrow$')
plt.title(r"Response for decaying sinusoid of frequency $2 \cdot 10^5$ Hz")
#plt.savefig("images/fig5",dpi=1000)
plt.show()

# Step response
plt.figure(6)
t = np.linspace(0,0.001,1000)
Vo = sp.step(H,T=t)
plt.plot(Vo[0],Vo[1])
plt.xlabel(r't$\rightarrow$')
plt.ylabel(r'$V_{o}\rightarrow$')
plt.title("Step Response of High Pass Butterworth Filter")
#plt.savefig("images/fig6",dpi=1000)
plt.show()