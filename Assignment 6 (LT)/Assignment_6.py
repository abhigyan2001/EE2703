import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from scipy.signal.ltisys import TransferFunctionContinuous as TransFun

#defining a new derived class from the Scipy Transfer Function class 
# with addition, division, multiplication and addition functionality
class LTIMore(TransFun):
    def __neg__(self):
        return LTIMore(-self.num,self.den)
    
    def __mul__(self,other):
        if type(other) in [int,float]:
            return LTIMore(self.num*other,self.den)
        elif type(other) in [TransFun,LTIMore]:
            n = np.polymul(other.num,self.num)
            d = np.polymul(other.den,self.den)
            return LTIMore(n,d)
    
    def __truediv__(self,other):
        if type(other) in [int,float]:
            return LTIMore(self.num,self.den*other)
        elif type(other) in [TransFun,LTIMore]:
            n = np.polymul(other.den,self.num)
            d = np.polymul(other.num,self.den)
            return LTIMore(n,d)
    
    def __rtruediv__(self,other):
        if type(other) in [int,float]:
            return LTIMore(self.den*other,self.num)
        elif type(other) in [TransFun,LTIMore]:
            n = np.polymul(self.den,other.num)
            d = np.polymul(self.num,other.den)
            return LTIMore(n,d)
    
    def __add__(self,other):
        if type(other) in [int,float]:
            return LTIMore(np.polyadd(self.num,other*self.den),self.den)
        elif type(other) in [TransFun,LTIMore]:
            n = np.polyadd(np.polymul(self.den,other.num),np.polymul(self.num,other.den))
            d = np.polymul(self.den,other.den)
            return LTIMore(n,d)
    
    def __sub__(self,other):
        return self + (-other)
    def __rsub__(self,other):
        return (-self) + other

def transfer(alpha,omega):
    return LTIMore([1,alpha],[1,2*alpha,omega**2+alpha**2])

def oneshotplot(i,x,y,xlabel='t',ylabel='x',title="Fig"):
    """To Plot Graphs in one single function"""
    plt.figure(i)
    plt.plot(x,y,"-")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig("images/fig"+str(i),dpi=1000)
    plt.show()

def main():
    # Question 1: Damped Springs
    F1 = transfer(0.5,1.5)
    X1 = LTIMore([1,0,2.25],[1])
    H1 = F1/X1
    times = np.linspace(0,50,1000)
    ts,xs = sp.impulse(H1,None,times)
    oneshotplot(1,ts,xs,'time','displacement',title="Plot for Question 1")

    # Question 2: Damped Spring, lower decay rate
    F2 = transfer(0.05,1.5)
    H2 = F2/X1
    ts2,xs2 = sp.impulse(H2,None,times)
    oneshotplot(2,ts2,xs2,'time','displacement',title="Plot for Question 2")

    # Question 3: Damped Springs at different frequencies
    myrange = np.arange(1.4,1.61,0.05)
    times1 = np.linspace(0,100,1000)
    plt.figure(3)
    for i,f in enumerate(myrange):
        currf = (np.cos(f*times1)*np.exp(-0.05*times1))*np.heaviside(times,0)
        _,y,_ = sp.lsim(H2,currf,times1)
        plt.subplot(3,2,i+1)
        plt.plot(times1,y,"-g")
    plt.suptitle("Impulse Response for different frequencies")
    plt.savefig("images/fig3.png",dpi=1000)
    plt.show()

    # Question 4: Coupled System
    times2 = np.linspace(0,20,1000)
    Hx = LTIMore([1,0,2],[1,0,3,0])
    Hy = LTIMore([2],[1,0,3,0])
    tsx,xsx = sp.impulse(Hx,None,times2)
    tsy,xsy = sp.impulse(Hy,None,times2)
    plt.figure(4)
    plt.plot(tsx,xsx,label='x')
    plt.plot(tsy,xsy,label='y')
    plt.xlabel("Time")
    plt.ylabel("x, y")
    plt.legend()
    plt.title("Coupled Oscillations")
    plt.savefig("images/fig4.png",dpi=1000)
    plt.show()

    # Question 5: 2 port network
    H2p = LTIMore([1000000],[0.000001,100,1000000])
    w,mag,phase = sp.bode(H2p)
    plt.figure(5)
    plt.subplot(2,1,1)
    plt.semilogx(w,mag)
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$|H(s)|$")
    plt.subplot(2,1,2)
    plt.semilogx(w,phase)
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\angle(H(s))$")
    plt.suptitle("Bode Plots for 2 Port Network")
    plt.savefig("images/fig5.png",dpi=1000)
    plt.show()

    # Question 6: 2 port network with custom input
    times3 = np.linspace(0,30*0.000001,1000)
    vi = np.multiply(np.cos(1000*times3)-np.cos(1000000*times3),np.heaviside(times3,0))
    _,y1,_ = sp.lsim(H2p,vi,times3)
    oneshotplot(6,times3,y1,'t',r'$v_{o}(t)$',r"Variation of $v_o(t)$ over 10$\mu$s")
    times4 = np.linspace(0,10*0.001,100000)
    vi = np.multiply(np.cos(1000*times4)-np.cos(1000000*times4),np.heaviside(times4,0))
    _,y2,_ = sp.lsim(H2p,vi,times4)
    oneshotplot(7,times4,y2,'t',r'$v_{o}(t)$',r"Variation of $v_o(t)$ over 1ms")

if __name__ == '__main__':
    main()