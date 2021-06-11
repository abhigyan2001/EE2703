import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
# Question 1:
## Working through the examples:

# Example 0:
## To test out the fft and ifft functions and check the differences and errors between them

x = np.random.rand(100)
X = np.fft.fft(x)
y = np.fft.ifft(X)
sidebyside = np.c_[x,y]
print(sidebyside)
print(abs(x-y).max())

# Example 1:
## To find the fourier spectrum of sin(5t)
x = np.linspace(0,2*np.pi,128)
y = np.sin(5*x)
Y = np.fft.fft(y)
plt.figure(0)
plt.subplot(2,1,1)
plt.plot(abs(Y),lw=1)
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of $\sin(5t)$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(np.unwrap(np.angle(Y)),lw=1)
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$k$",size=16)
plt.grid(True)
plt.savefig("images/fig0.png")
plt.show()

# Creating a helper function to aid in plotting.
# The template is based on the examples in the assignment PDF
def plotter(function, start, end, samples, lim, title, sig=False, save=True, fignum=1):
    """
    Plots the given function string within range [start,end) with given number of samples
    --Arguments--
    function:   a python expression for the function to be evaluated in terms of 't'
    start:      the value to start sampling at
    end:        the value to end sampling at
    samples:    the number of samples to be used for t
    lim:        the plot will be plotted between [-lim,lim]
    title:      the title of the plot in r-string form
    sig=False:  whether to plot phase points having significant phase only or all. Plots all if not specified
    save=False: boolean, saves figure as "fig"+str(fignum)+".png" if True, doesn't save if not specified
    fignum=1:   the figure number for matplotlib. Will be used to save figure if save=True

    --Example usage--
    plotter('np.sin(6*t)', -2*np.pi, 2*np.pi, 128, 10, r'Corrected Spectrum of $\sin(6t)$',True)
    """
    t = np.linspace(start,end,samples+1)[:-1]

    y = eval(function)
    Y = np.fft.fftshift(np.fft.fft(y))/samples
    # We need to ensure that the spacing and the sampling frequency correspond to real frequencies, 
    # hence we divide the number of samples by the number of multiples of pi in our range so that 
    # the frequencies we get correspond to real frequency values
    maxval = samples/((end-start)//np.pi)
    w = np.linspace(-maxval,maxval,samples+1)[:-1]
    plt.figure(fignum)
    plt.subplot(2,1,1)
    plt.plot(w,abs(Y),lw=1)
    plt.xlim([-lim,lim])
    plt.ylabel(r"$|Y|$",size=16)
    plt.title(title)
    plt.grid(True)
    plt.subplot(2,1,2)
    if not sig:
        plt.plot(w,np.angle(Y),'ro',lw=1)
    ii = np.where(abs(Y)>1e-3)
    plt.plot(w[ii],np.angle(Y[ii]),'go',lw=1)
    plt.xlim([-lim,lim])
    plt.ylim([-(np.pi+0.5),(np.pi+0.5)])
    plt.ylabel(r"Phase of $Y$",size=16)
    plt.xlabel(r"$\omega$",size=16)
    plt.grid(True)
    if save:
        plt.savefig("images/fig"+str(fignum)+".png")
    plt.show()

# Example 1 with corrections and fftshift:
## To find the fourier spectrum of sin(5t) with the peaks in the right locations
## This time, we use the helper function which takes fftshift and magnitude scaling into account
plotter("np.sin(5*t)", 0, 2*np.pi, 128, 10, r"Corrected Spectrum of $\sin(5t)$",save=True, fignum=1)

# Example 2:
## To find the Fourier Spectrum of the Amplitude Modulated Function (1+0.1cos(t))cos(10t)
plotter("(1 + 0.1*np.cos(t))*np.cos(10*t)",0,2*np.pi,128,15,r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$",save=True,fignum=2)

# Example 2 with corrections:
## To find the Fourier Spectrum of the Amplitude Modulated Function (1+0.1cos(t))cos(10t)
## Now with 512 samples
### Note that here we are sampling times from -4pi to 4pi.
plotter("(1 + 0.1*np.cos(t))*np.cos(10*t)",-4*np.pi,4*np.pi,512,15,r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$",save=True,fignum=3)

### Now, we are sampling times from 0 to 8pi.
plotter("(1 + 0.1*np.cos(t))*np.cos(10*t)",0,8*np.pi,512,15,r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$",save=True,fignum=4)

### As we can see, there is a slight difference between the two phase plots 
### while the magnitude plots are identical. 
### This is just an error due to the small non-zero values in the DFT. 
### Phase is undefined if magnitude is zero, hence we can safely ignore those values

# Question 2:

# Spectrum of sin^3(t):

plotter("(np.sin(t))**3",0,8*np.pi,512,4,r"Spectrum of $\sin^3(t)$",save=True,fignum=5)

## Expected Spectrum:
# Seeing as $sin(3t)=3sin(t)−4\sin^3(t)$, it follows $sin^3(t)=\frac{3}{4}sin(t)−\frac{1}{4}sin(3t)$ 
# hence we should be having peaks of 0.375 at $t = \pm 1$ and peaks of 0.125 at $t = \pm 3$. 
# This is exactly what we see in the magnitude spectrum.

# Also, as it is in terms of sin(x), the phase spectrum is odd and is $-\frac{\pi}{2}$ at $t=-3$ and $1$
# while it is $\frac{\pi}{2}$ at $t=-1$ and $3$

# Spectrum of cos^3(t):

plotter("(np.cos(t))**3",0,8*np.pi,512,4,r"Spectrum of $\cos^3(t)$",save=True,fignum=6)

# Seeing as cos(3t)=4\cos^3(t)-3cos(t), it follows cos^3(t)=\frac{3}{4}cos(t)+\frac{1}{4}cos(3t) 
# hence we should be having peaks of 0.375 at $t = \pm 1$ and peaks of 0.125 at $t = \pm 3$. 
# This is exactly what we see in the magnitude spectrum.

# Also, as it is in terms of cos(t), the phase spectrum is 0 everywhere.

# Question 3:

## Spectrum of cos(20t + 5cos(t))
## With significant points in phase plot only

plotter("np.cos(20*t + 5*np.cos(t))",0,8*np.pi,512,40,r"Spectrum of $\cos(20t + 5\cos(t))$",sig=True,save=True,fignum=7)

# Firstly, wow! That plot looks awesome! Now, I need to really study and figure out why it came like that haha
# A first guess would be that since we have a +5cos(t) term internally, it adds some extra frequencies near 20 to the Fourier Transform
# which dies out as we move away from 20 on either side. 

# The phase plot also confirms this, in a way, as we can see that at the frequencies where the magnitude is significant,
# the phases oscillate quickly around 20 between $\frac{\pi}{2}$ and $\pi$ on the right side, and between $\pi$ and $-\frac{\pi}{2}$ on the left
# and then the rate of oscillation decreases as we move away from 20

# Question 4:

## Spectrum of a Gaussian: $\exp(-t^2/2)$

### Note that we need to sample this over a range that is symmetric, as the function is not periodic, 
### and we will get the wrong phase spectrum if we sample this over, say, $[0,8\pi)$ as seen below:
plotter("np.exp(-(t**2)/2)",0*np.pi,8*np.pi,512,50,r"Spectrum of a Gaussian $\exp(\frac{-t^2}{2})$",sig=False,save=True,fignum=8)
### Along similar lines, since we are sampling using linspace and then discarding one value, the space is not centered at 0.
### Hence, for this one, we will plot it without the helper function
### The DFT of the gaussian is very stubborn and doesn't come out to be the same as the CFT of the gaussian due to sampling problems
### Hence, we need to take the absolute value of the function before plotting it so as to get rid of the negative values that tend to arise

### We will be using 4194305 samples over -8pi to 8pi to get an error value less than 1e-6
t = np.linspace(-8*np.pi,8*np.pi,4194305)
y = np.exp(-(t**2)/2)
Y_init = np.fft.fftshift(abs(np.fft.fft(y)))/4194305
# Normalized value of Y
Y = Y_init*np.sqrt(2*np.pi)/max(Y_init)
w = np.linspace(-262144,262144,4194305)

Y_actual = np.sqrt(2*np.pi)*np.exp((-w**2)/2)

print("Error in Gaussian = "+str(max(abs(Y_actual-Y))))
plt.figure(9)
plt.subplot(2,1,1)
plt.plot(w,abs(Y),lw=1)
plt.xlim([-50,50])
plt.ylabel(r"$|Y|$",size=16)
plt.title(r"Spectrum of a Gaussian $\exp(\frac{-t^2}{2})$")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro',lw=1)
ii = np.where(abs(Y)>1e-3)
plt.plot(w[ii],np.angle(Y[ii]),'go',lw=1)
plt.xlim([-50,50])
plt.ylim([-(np.pi+0.5),(np.pi+0.5)])
plt.ylabel(r"Phase of $Y$",size=16)
plt.xlabel(r"$\omega$",size=16)
plt.grid(True)
plt.savefig("images/fig9.png")
plt.show()