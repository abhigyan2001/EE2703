import sys
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate as tb

def printhelp(err=False):
    if err:
        print("Error: Invalid inputs")
    print("n    (int)   = spatial grid size")
    print("M    (int)   = number of electrons injected per turn")
    print("Msig (int)   = standard deviation for electrons injected per turn")
    print("nk   (int)   = number of turns to simulate")
    print("u0   (float) = threshold velocity required to excite an atom")
    print("p    (float) = probability that a collision results in an excited atom")
    print("\nUse as: python Assignment_6.py n M Msig nk u0 p")
    exit()

def getparams():
    #Setting up Parameters with default values
    n    = 100  #spatial grid size
    M    = 5    #number of electrons injected per turn
    Msig = 1    #standard deviation for electrons injected per turn
    nk   = 500  #number of turns to simulate
    u0   = 5    #threshold velocity required to excite an atom
    p    = 0.25 #probability that a collision results in an excited atom

    if len(sys.argv)==1:
        print("Default parameters used: n=100, M=5, Msig=1, nk=500, u0=5, p=0.25\n")
        print("To use your own parameters, use as:\npython Assignment_6.py n M nk u0 p\n(Note that all parameters must be entered)")
        print("\nFor more information, use python Assignment_6 help")
    elif len(sys.argv)==7:
        try:
            n    = int(sys.argv[1])
            M    = int(sys.argv[2])
            Msig = int(sys.argv[3])
            nk   = int(sys.argv[4])
            u0   = float(sys.argv[5])
            p    = float(sys.argv[6])
        except:
            printhelp(True)
        if p>1:
            print("Error: p must be less than 1")
            printhelp()
            sys.exit()
    elif len(sys.argv)==2:
        if sys.argv[1]=='help':
            printhelp()
        else:
            printhelp(True)
    else:
        printhelp(True)
    return n,M,Msig,nk,u0,p

n,M,Msig,nk,u0,p = getparams()

def montecarlo(n,M,Msig,nk,u0,p,accurateMode=False):
    xx = np.zeros((n*M),dtype=float)
    u  = xx.copy()
    dx = xx.copy()

    I = []
    X = []
    V = []

    for k in range(nk):
        # Step 1: Advancing the electrons
        # Step 1.1: Finding the locations of the active electrons
        ii = np.where(xx>0)
        # Step 1.2: Adding all the electron positions to the X and V vectors
        X.extend(xx[ii].tolist())
        V.extend( u[ii].tolist())
        # Step 1.3: Adjusting the displacement, position and velocity using the equations of motion
        dx[ii]  =  u[ii] + 0.5
        xx[ii] += dx[ii]
        u[ii]  += 1
        # Step 1.4: Setting the electrons that reached the anode to inactive
        u[np.where(xx>=n)] = 0
        xx[np.where(xx>=n)] = 0
        
        # Step 2: Handling collisions and photon generation
        # Step 2.1: Select all electrons with velocity greater than threshold velocity
        kk = np.where(u>=u0)[0]
        # Step 2.2: Generate some electrons which excite the atoms
        ll = np.where(np.random.rand(len(kk))<=p)[0]
        kl = kk[ll]
        if accurateMode:
            # Step 2.3: Don't set the velocity of the electrons that collided to 0
            #u[kl] = 0
            # Step 2.4: Find the position of collision of these electrons
            dt = np.random.rand(len(kl))
            xx[kl]=xx[kl]-dx[kl]+((u[kl]-1)*dt+0.5*dt*dt)+0.5*(1-dt)**2 #accurate collision location
            u[kl]=1-dt #accurate velocity update
        else:
            # Step 2.3: Set the velocity of the electrons that collided to 0
            u[kl] = 0
            # Step 2.4: Find the position of collision of these electrons
            xx[kl]=xx[kl]-dx[kl]*np.random.rand(len(dx[kl]))
        
        # Step 2.5: Add all the excited atom locations to the list of photon positions
        I.extend(xx[kl].tolist())

        # Step 3: Injecting new electrons
        # Step 3.1: Calculate number of electrons based on random number generator
        m = (np.random.randn())*Msig+M
        # Step 3.2: Choose min of free slots available and number of electrons from above
        freeslots = np.where(xx==0)[0]
        m = int(min(m,len(freeslots)))
        # Step 3.3: Set first m freeslot locations to 1
        xx[freeslots[:m]] = 1
    return I,X,V
    
I,X,V = montecarlo(n,M,Msig,nk,u0,p)

# Plotting the graphs
plt.figure(0)
counts1,bins1,_=plt.hist(X,n,edgecolor='black',linewidth=0.5)
plt.title("Population Plot of Electron Position")
plt.xlabel(r"$x\rightarrow$")
plt.ylabel(r"Number of Electrons$\rightarrow$")
plt.savefig("images/fig0.png",dpi=1000)
plt.show()

plt.figure(1)
counts,bins,_ = plt.hist(I,n,[0,n],edgecolor='black',linewidth=0.5)
plt.title("Population Plot of Intensity")
plt.xlabel(r"$x\rightarrow$")
plt.ylabel(r"Number of Photons Emitted ($\propto$ Intensity)$\rightarrow$")
plt.savefig("images/fig1.png",dpi=1000)
plt.show()

plt.figure(2)
plt.scatter(X,V,marker='^')
plt.title("Electron Phase Space")
plt.xlabel("$x$")
plt.ylabel("$v$")
plt.savefig("images/fig2.png",dpi=1000)
plt.show()

# Tabulating the data
xpos = 0.5*(bins[0:-1]+bins[1:])
print("Intensity data:")
print(tb(np.c_[xpos,counts],headers=['xpos','count']))

## For More Accurate Updates
I,X,V = montecarlo(n,M,Msig,nk,u0,p,True)

# Plotting the more accurately updated graphs
plt.figure(0)
counts1,bins1,_=plt.hist(X,n,edgecolor='black',linewidth=0.5)
plt.title("Population Plot of Electron Position")
plt.xlabel(r"$x\rightarrow$")
plt.ylabel(r"Number of Electrons$\rightarrow$")
plt.savefig("images/fig0_acc.png",dpi=1000)
plt.show()

plt.figure(1)
counts,bins,_ = plt.hist(I,n,[0,n],edgecolor='black',linewidth=0.5)
plt.title("Population Plot of Intensity")
plt.xlabel(r"$x\rightarrow$")
plt.ylabel(r"Number of Photons Emitted ($\propto$ Intensity)$\rightarrow$")
plt.savefig("images/fig1_acc.png",dpi=1000)
plt.show()

plt.figure(2)
plt.scatter(X,V,marker='^',c='',edgecolor='red',linewidth=0.2)
plt.title("Electron Phase Space")
plt.xlabel("$x$")
plt.ylabel("$v$")
plt.savefig("images/fig2_acc.png",dpi=1000)
plt.show()

# Tabulating the more accurate data
xpos = 0.5*(bins[0:-1]+bins[1:])
print("Intensity data:")
print(tb(np.c_[xpos,counts],headers=['xpos','count']))