import numpy as np
import matplotlib.pyplot as plt
from numpy.fft.helper import fftshift
from numpy.lib.function_base import meshgrid
plt.style.use("seaborn")
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

## Pre-requisite work:
# Setting up a plotting function

def plot_spectrum(w,Y,num,funcnm,type='log',xlims=[-10,10],ylims=None,save=True):
    plt.figure(num)
    plt.subplot(2,1,1)
    if type=='log':
        plt.semilogy(w,abs(Y),lw=2)
    elif type=='lin':
        plt.plot(w,abs(Y),lw=2)
    elif type=='linpts':
        plt.plot(w,abs(Y),lw=2)
        plt.plot(w,abs(Y),'o',lw=2)
    plt.xlim(xlims)
    if ylims:
        plt.ylim(ylims)
    plt.ylabel(r"$|Y|$",size=16)
    plt.title(f"Spectrum of {funcnm}")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(w,np.angle(Y),'ro',lw=2)
    plt.xlim(xlims)
    if ylims:
        plt.ylim(ylims)
    plt.ylabel(r"Phase of $Y$",size=16)
    plt.xlabel(r"$\omega$",size=16)
    plt.grid(True)
    if save:
        plt.savefig(f"images/fig{num}.png")
    plt.show()

def plot_func(ts,fs,num,funcnm,marker=None,save=True):
    plt.figure(num)
    for t,f in zip(ts,fs):
        if marker:
            plt.plot(t,f,f'{marker}',lw=2)
        else:
            plt.plot(t,f,lw=2)
    plt.ylabel(r"$y$",size=16)
    plt.xlabel(r"$t$",size=16)
    plt.title(funcnm)
    plt.grid(True)
    if save:
        plt.savefig(f"images/fig{num}.png")
    plt.show()

## Question 1: Working through given examples:

ctr = 1
t=np.linspace(-np.pi,np.pi,65)[:-1]
fmax=1/(t[1]-t[0])
y=np.sin(t)
y[0]=0 # the sample corresponding to -tmax should be set zero
y=np.fft.fftshift(y) # make y start with y(t=0)
Y=np.fft.fftshift(np.fft.fft(y))/64.0
w=np.linspace(-np.pi*fmax,np.pi*fmax,65)[:-1]
plot_spectrum(w,Y,ctr,r"$\sin\left(t\right)$")
ctr+=1


y=(np.sin(np.sqrt(2)*t))
y[0]=0 # the sample corresponding to -tmax should be set zero
y=np.fft.fftshift(y) # make y start with y(t=0)
Y=np.fft.fftshift(np.fft.fft(y))/64.0
plot_spectrum(w,Y,ctr,r"$\sin\left(\sqrt{2}t\right)}$")
ctr+=1

t1=np.copy(t)
t2=np.linspace(-3*np.pi,-np.pi,65)[:-1]
t3=np.linspace(np.pi,3*np.pi,65)[:-1]
# y=sin(sqrt(2)*t)
plot_func([t1,t2,t3],[np.sin(np.sqrt(2)*t1),np.sin(np.sqrt(2)*t2),np.sin(np.sqrt(2)*t3)],ctr,r"$\sin\left(\sqrt{2}t\right)$")
ctr+=1

y=np.sin(np.sqrt(2)*t1)
plot_func([t1,t2,t3],[y,y,y],ctr,r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$",marker='o')
ctr+=1

t=np.linspace(-np.pi,np.pi,65)[:-1]
fmax=1/(t[1]-t[0])
y=t
y[0]=0 # the sample corresponding to -tmax should be set zero
y=np.fft.fftshift(y) # make y start with y(t=0)
Y=np.fft.fftshift(np.fft.fft(y))/64.0
w=np.linspace(-np.pi*fmax,np.pi*fmax,65)[:-1]

plt.figure(ctr)
plt.semilogx(abs(w),20*np.log10(abs(Y)),lw=2)
plt.xlim([1,10])
plt.ylim([-20,0])
plt.xticks([1,2,5,10],["1","2","5","10"],size=16)
plt.ylabel(r"$|Y|$ (dB)",size=16)
plt.title(r"Spectrum of a digital ramp")
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig(f"images/fig{ctr}.png")
plt.show()
ctr+=1

n=np.arange(64)
wnd=np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/63))
y=np.sin(np.sqrt(2)*t1)*wnd
plot_func([t1,t2,t3],[y,y,y],ctr,r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$",marker='o')
ctr+=1

t=np.linspace(-np.pi,np.pi,65)[:-1]
fmax=1/(t[1]-t[0])
n=np.arange(64)
wnd=np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/63))
y=np.sin(np.sqrt(2)*t)*wnd
y[0]=0 # the sample corresponding to -tmax should be set zero
y=np.fft.fftshift(y) # make y start with y(t=0)
Y=np.fft.fftshift(np.fft.fft(y))/64.0
w=np.linspace(-np.pi*fmax,np.pi*fmax,65)[:-1]
plot_spectrum(w,Y,ctr,r"$\sin\left(\sqrt{2}t\right)\times w(t)$",type='lin',xlims=[-8,8])
ctr+=1

t=np.linspace(-4*np.pi,4*np.pi,257)[:-1]
fmax=1/(t[1]-t[0])
n=np.arange(256)
wnd=np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/256))
y=np.sin(np.sqrt(2)*t)
#y=np.sin(1.25*t)
y=y*wnd
y[0]=0 # the sample corresponding to -tmax should be set zero
y=np.fft.fftshift(y) # make y start with y(t=0)
Y=np.fft.fftshift(np.fft.fft(y))/256.0
w=np.linspace(-np.pi*fmax,np.pi*fmax,257)[:-1]
plot_spectrum(w,Y,ctr,r"$\sin\left(\sqrt{2}t\right)\times w(t)$",type='linpts',xlims=[-4,4])
ctr+=1

## Question 2

t=np.linspace(-4*np.pi,4*np.pi,257)[:-1]
fmax=1/(t[1]-t[0])
n=np.arange(256)
wnd=np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/256))
y1=np.cos(0.86*t)**3
y2=y1*wnd

y1[0]=0 # the sample corresponding to -tmax should be set zero
y2[0]=0 # the sample corresponding to -tmax should be set zero
y1=np.fft.fftshift(y1) # make y start with y(t=0)
y2=np.fft.fftshift(y2) # make y start with y(t=0)

Y1=np.fft.fftshift(np.fft.fft(y1))/256.0
Y2=np.fft.fftshift(np.fft.fft(y2))/256.0
w=np.linspace(-np.pi*fmax,np.pi*fmax,257)[:-1]

plot_spectrum(w,Y1,ctr,r"$\cos^3\left(0.86t\right)$",type='linpts')
ctr+=1
plot_spectrum(w,Y2,ctr,r"$\cos^3\left(0.86t\right) \times w(t)$",type='linpts')
ctr+=1

## Question 3

def om_del(func,ts,pow=1.7):
    N = len(ts)
    fmax = 1/(ts[1]-ts[0])
    w = np.linspace(-np.pi*fmax,np.pi*fmax,N+1)[:-1]
    y = func
    n = np.arange(N)
    wnd = fftshift(0.54+0.46*np.cos(2*np.pi*n/(N-1)))
    y = y*wnd
    y[0] = 0
    Y = np.fft.fftshift(np.fft.fft(np.fft.fftshift(y)))/N
    delta = np.angle(Y[::-1][np.argmax(abs(Y[::-1]))])
    omega = np.sum(abs(Y**pow*w))/np.sum(abs(Y)**pow)
    return omega,delta

# The peaks at omega_0 can be found using the expectation value
# and the value of delta can be found as the angle of that omega_0

omega = 0.5
delta = np.pi

# Now taking the fourier transform of cos(omega * t + delta)

t = np.linspace(-np.pi,np.pi,129)[:-1]
fmax = 1/(t[1]-t[0])
y1 = np.cos(omega*t + delta)
om,delt = om_del(y1,t,1.7)
print(f"The estimated value of omega is {om} and delta is {delt}")

## Question 4

y2 = np.cos(omega*t + delta) + 0.1*np.random.randn(128)
om,delt = om_del(y2,t,2.4)
print(f"The estimated value of omega is {om} and delta is {delt}")

n = np.arange(128)
wnd = np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/128))
y11 = wnd*y1
y11[0]=0
y11 = np.fft.fftshift(y11)
Y = np.fft.fftshift(np.fft.fft(y11))/128
w = np.linspace(-np.pi*fmax,np.pi*fmax,129)[:-1]
plot_spectrum(w,Y,ctr,rf"$\cos\left({omega}t+{delta}\right)$")
ctr+=1
plt.show()

## Question 5
t = np.linspace(-np.pi,np.pi,1025)[:-1]
fmax = 1/(t[1]-t[0])
y = np.cos(16*t*(1.5+t/(2*np.pi)))
y = np.fft.fftshift(y)
Y = np.fft.fftshift(np.fft.fft(y))/1024
w = np.linspace(-np.pi*fmax,np.pi*fmax,1025)[:-1]
plot_spectrum(w,Y,ctr,r"$\cos\left(16t\left(1.5+\frac{t}{2\pi}\right)\right)$",type='linpts',xlims=[-50,50])
ctr+=1

y = np.cos(16*t*(1.5+t/(2*np.pi)))
plot_func([t], [y], ctr, r"$\cos\left(16t\left(1.5+\frac{t}{2\pi}\right)\right)$",save=True)
ctr+=1

## Question 6

import mpl_toolkits.mplot3d.axes3d as ax3d

Ys = []
for i in range(16):
    tlow = np.pi*(-1+i/8)
    thigh = tlow+np.pi/8
    t = np.linspace(tlow,thigh,65)[:-1]
    y = np.fft.fftshift(np.cos(16*t*(1.5+t/(2*np.pi))))
    Y = np.fft.fftshift(np.fft.fft(y))/64
    Ys.append(Y)

Ys = np.asarray(Ys)
t1 = np.linspace(-np.pi,np.pi,16)
ts = np.linspace(-np.pi,np.pi,1025)[:-1]
fmax = 1/(ts[1]-ts[0])
w = np.linspace(-np.pi*fmax,np.pi*fmax,65)[:-1]
ax = ax3d.Axes3D(plt.figure(ctr))
ctr+=1
Ys1 = Ys.copy()
ii = np.where(abs(w)>150)
Ys1[:,ii]=np.NaN
t1,w = np.meshgrid(t1,w)
surface = ax.plot_surface(t1,w,abs(Ys1).T,rstride=1,cstride=1,cmap=plt.get_cmap("jet"))
plt.ylabel(r'$\omega\rightarrow$',size=16)
plt.xlabel(r'$t\rightarrow$',size=16)
ax.set_ylim([-150,150])
ax.set_zlabel(r'$|Y|$')
plt.show()