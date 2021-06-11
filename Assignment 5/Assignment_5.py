import sys
from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import matplotlib.pyplot as plt

Nx = 25       # size along x
Ny = 25       # size along y
radius = 8    # radius of central lead
Niter = 1500  # number of iterations to perform

if len(sys.argv) == 5:
    Nx = int(sys.argv[1])
    Ny = int(sys.argv[2])
    radius = int(sys.argv[3])
    Niter = int(sys.argv[4])

elif len(sys.argv) == 1:
    print("Alternate Usage: python Assignment_5.py Nx Ny radius Niter")

else:
    print("Wrong number of parameters entered.\nPlease use as: python Assignment_5.py Nx Ny radius Niter")
    exit()
    
#Setting up the Plate to fill in with zeros

phi = np.zeros((Ny, Nx), dtype=float)
x = np.linspace(-Nx/2,Nx/2,Nx)
y = np.linspace(-Ny/2,Ny/2,Ny)
X, Y = np.meshgrid(x, y)

#Setting voltage to 1 in the circle of given radius in contact with electrode
volt1Nodes = np.where(np.square(X)+np.square(Y) <= radius**2)
phi[volt1Nodes] = 1.0

#Plotting initial figure before iterating
plt.figure(1)
plt.gca().set_aspect('equal', adjustable='box')
contPlot = plt.contourf(X, Y, phi, 50, cmap=cm.hot) # we use the "hot" color map
plt.colorbar()
v1Plot = plt.scatter(x[volt1Nodes[0]], y[volt1Nodes[1]], color='r', label='Vo = 1V')
plt.title(r'Contour Plot of $\phi$ before iterating (in V)')
plt.legend()
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
#plt.savefig("images/fig1.png",dpi=1000)
plt.show()

#Solving the Laplace Equation Iteratively

errors = np.zeros(Niter)
for k in range(Niter):
    #save phi
    oldphi = phi.copy()                         
	
    #update phi
    phi[1:-1,1:-1]=0.25*(oldphi[1:-1,0:-2]+ oldphi[1:-1,2:]+ oldphi[0:-2,1:-1]+ oldphi[2:,1:-1])
    
    #boundary conditions
    phi[1:-1, 0]=phi[1:-1, 1]             #left edge
    phi[1:-1,-1]=phi[1:-1,-2]             #right edge
    phi[-1,1:-1]=phi[-2,1:-1]		      #top edge
    phi[ 0,1:-1]=0                        #bottom edge  
    
    #corner update
    phi[ 0, 0] = 0.5*(phi[ 0, 1] + phi[ 1, 0])
    phi[ 0,-1] = 0.5*(phi[ 0,-2] + phi[ 1,-1])
    phi[-1, 0] = 0.5*(phi[-1, 1] + phi[-2, 0])
    phi[-1,-1] = 0.5*(phi[-1,-2] + phi[-2,-1])
    
    #Set the voltage = 1 in region in contact with electrode
    phi[volt1Nodes] = 1.0
    
    #errors
    errors[k]=(abs(phi-oldphi)).max()
    

#Using lstsq to find the variation of error with iterations
iterations = np.asarray(range(Niter))
iter50 = iterations[::50]
log_err = np.log(errors[0::50])

M = np.c_[iter50,np.ones(Niter//50)]
p = np.linalg.lstsq(M,log_err,rcond=None)[0]
prederror = np.exp(p[1]+p[0]*iter50)

M500 = np.c_[iterations[500::50],np.ones((Niter - 500)//50)]
p500 = np.linalg.lstsq(M500,log_err[10::],rcond=None)[0]
prederror500 = np.exp(p500[1]+p500[0]*iter50[10:])

# semilog plot        
plt.figure(2)
plt.semilogy(iter50, errors[0::50], 'ro',label='Samples every 50 iterations')
plt.semilogy(iterations, errors, label='Actual Errors')
plt.semilogy(iter50, prederror, label='Linear fit for all iterations')
plt.semilogy(iter50[10:], prederror500, label='Linear fit for iterations > 500')
plt.title('Semilog Plot of error')
plt.xlabel(r'iterations$\rightarrow$')
plt.ylabel(r'error$\rightarrow$')
plt.legend()
#plt.savefig("images/fig2.png",dpi=1000)
plt.show()

#loglog plot
plt.figure(3)
plt.loglog(iterations, errors, label='Error (Log-Log)')
plt.loglog(iter50, errors[0::50], 'ro',label='Samples every 50 iterations')
plt.title('Loglog Plot of error')
plt.xlabel(r'iterations$\rightarrow$')
plt.ylabel(r'error$\rightarrow$')
plt.legend()
#plt.savefig("images/fig3.png",dpi=1000)
plt.show()

#Question3
fig = plt.figure(4)
# open a new figure
ax = p3.Axes3D(fig)
ax.set_xlabel('X')
ax.set_ylabel('Y')
surf = ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap=cm.jet)
ax.set_title(r"3-D surface plot of $\phi(x,y)$")
fig.colorbar(surf,shrink=0.5)
#plt.savefig("images/fig4.png",dpi=1000)
plt.show()

plt.figure(5)
plt.gca().set_aspect('equal', adjustable='box')
plt.contourf(X, Y, phi, 50, cmap=cm.hot)
plt.colorbar()
plt.scatter(x[volt1Nodes[0]], y[volt1Nodes[1]], color='r', label = 'Vo = 1V')
plt.legend(loc='upper right')
plt.title(r'Contour Plot of final $\phi$ after iterating (in V)')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
#plt.savefig("images/fig5.png",dpi=1000)
plt.show()

#Question4

#Jx and Jy
Jx = np.zeros((Ny, Nx), dtype=float)
Jy = np.zeros((Ny, Nx), dtype=float)
Jx = 0.5 * (phi[1:-1,0:-2]-phi[1:-1,2:])
Jy = 0.5 * (phi[ :-2,1:-1]-phi[2:,1:-1])

plt.figure(6)
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(x[volt1Nodes[0]], y[volt1Nodes[1]], color='r', label = 'Vo = 1V')
plt.quiver(X[1:-1,1:-1],Y[1:-1,1:-1],Jx,Jy,scale=4,label=r'$\vec{J}$')
plt.xlabel(r"$X$")
plt.ylabel(r"$Y$")
plt.legend(loc='upper right')
plt.title("Current Density Vector")
#plt.savefig("images/fig6.png",dpi=1000)
plt.show()


temp = 300*np.ones((Ny,Nx),dtype=float)
for k in range(Niter):
    #save temp
    oldtemp = temp.copy()                         
	
    #update temp
    temp[1:-1,1:-1]=0.25*(oldtemp[1:-1,0:-2]+ oldtemp[1:-1,2:]+ oldtemp[0:-2,1:-1]+ oldtemp[2:,1:-1] + Jx**2 + Jy**2)
    
    #boundary conditions
    temp[1:-1, 0]=temp[1:-1, 1]             #left edge
    temp[1:-1,-1]=temp[1:-1,-2]             #right edge
    temp[-1,1:-1]=temp[-2,1:-1]		        #top edge
    temp[ 0,1:-1]=300                       #bottom edge  
    
    #corner update
    temp[ 0, 0] = 0.5*(temp[ 0, 1] + temp[ 1, 0])
    temp[ 0,-1] = 0.5*(temp[ 0,-2] + temp[ 1,-1])
    temp[-1, 0] = 0.5*(temp[-1, 1] + temp[-2, 0])
    temp[-1,-1] = 0.5*(temp[-1,-2] + temp[-2,-1])
    
    #Set the temp = 300 in region in contact with electrode
    temp[volt1Nodes] = 300
    
#Plotting the temperatures
plt.figure(7)
plt.gca().set_aspect('equal', adjustable='box')
plt.contourf(X, Y, temp, 50, cmap=cm.hot)
plt.colorbar()
plt.scatter(x[volt1Nodes[0]], y[volt1Nodes[1]], color='r', label = 'T = 300K')
plt.legend(loc='upper right')
plt.title(r'Contour Plot of final Temperature (in K)')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.savefig("images/fig7.png",dpi=1000)
plt.show()

plt.figure(8)
plt.gca().set_aspect('equal', adjustable='box')
plt.contourf(X, Y, temp, 50, cmap=cm.hot)
plt.colorbar()
plt.scatter(x[volt1Nodes[0]], y[volt1Nodes[1]], color='r', label = 'T = 300K')
plt.quiver(X[1:-1,1:-1],Y[1:-1,1:-1],Jx,Jy,scale=4,label=r'$\vec{J}$')
plt.legend(loc='upper right')
plt.title(r'Contour Plot with Quiver Plot')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.savefig("images/fig8.png",dpi=1000)
plt.show()

