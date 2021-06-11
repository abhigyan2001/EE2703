
import numpy as np
import matplotlib.pyplot as plt

# Question 2: Breaking the Volume into a 3x3x1000 mesh, separated by 1cm each

x = np.linspace(-0.99,1.01,3,dtype=np.longdouble)
y = np.linspace(-1.01,0.99,3,dtype=np.longdouble)
z = np.linspace(1,1000,1000,dtype=np.longdouble)

X,Y,Z = np.meshgrid(x,y,z,indexing='ij') # meshgrid returns a set of 3 3D arrays that helps to run 
# the functions over the entire grid without having to index each of them individually

# Question 3: 
# Part 1: Generating the points on the loop antenna

a = 10 # radius of the loop
N = 100  # number of loop sections
k = 0.1  # k = w/c = 1/10 = 0.1 rad/cm
phi = np.linspace(0,2*np.pi,N+1,dtype=np.longdouble)[:-1]
exs = a*np.cos(phi) # x values
wys = a*np.sin(phi) # y values

# Question 4: The r' vector

r_prime = np.c_[exs,wys] # converting the 2 individual arrays to a 100x2 array

# Question 3 contd.
# Part 2: Finding the current elements on the loop
# Now finding the current magnitudes and directions at these points

curr_mag = 1e7*np.cos(phi)

# Question 4 contd.
# Obtaining dl' = [-r*d(phi)sin(phi)x-hat, r*d(phi)cos(phi)y-hat]:
# Hence dl' = d(phi)*[-y,x] for each [x,y], and d(phi)=2pi/100

dl_prime = 2*np.pi/N * np.asarray([-wys,exs],dtype=np.longdouble).T

# Current will be curr_mag * dl

plt.figure(1)
plt.scatter(*(r_prime.T))
plt.quiver(*(r_prime.T),*(curr_mag*dl_prime.T*N/(2*np.pi)))
plt.gca().set_aspect('equal') # plots distances equally on both axes
plt.grid(True)
plt.title('Current Elements on Antenna Loop')
plt.xlabel(r'$x\rightarrow$')
plt.ylabel(r'$y\rightarrow$')
plt.savefig('images/fig1.png',dpi=1000)
plt.show()

# Question 5

r = np.array((X,Y,Z))

def calc(l):
    r_prime_wz = np.c_[r_prime,np.zeros([N,1])]
    # adding the z coordinate, which is a row of zeros
    # r_prime_wz stands for "r prime with zeros"
    r1 = (np.tile(r,(N,1,1,1)).reshape((N,3,3,3,1000)))
    # r1 is a rearranged and repeated version of r
    # First, we tile r 100 times so that we have 
    # 100 different copies of it to handle each of the values of l
    # Then we reshape it into a 5D array with 100 copies of the X,Y,Z arrays, 
    # each of which are 3,3,1000 arrays. Hence, its shape is (100,3,3,3,1000)
    R = r1 - r_prime_wz.reshape((N,3,1,1,1))
    # Before we start subtracting, we reshape r prime with zeros
    # to a (100,3,1,1,1) array so that it can be subtracted from 
    # r1 using numpy broadcasting
    return np.linalg.norm(R,axis=1)[l]
    # Finally, we take the norm of it along the first axis 
    # (i.e. the axis that has the individual X,Y,Z components)
    # And therefore we get a 100,3,3,1000 array, and return its l'th element.
    # This is the set of distances of the lth section of the 
    # loop from all the 3x3x1000 points in space

# Question 6

def calc(l=None):
    r_prime_wz = np.c_[r_prime,np.zeros([N,1])]
    r1 = (np.tile(r,(N,1,1,1)).reshape((N,3,3,3,1000)))
    R = r1 - r_prime_wz.reshape((N,3,1,1,1))
    modR = np.linalg.norm(R,axis=1)
    # Same as earlier version till here
    tempval = curr_mag.reshape(N,1,1,1)/(1e7)*np.exp(-1j*k*modR)/modR
    Aijkl = np.asarray([tempval*dl_prime[:,0].reshape(N,1,1,1),tempval*dl_prime[:,1].reshape(N,1,1,1)])
    # The Aijkl value as per equation 1
    return Aijkl[l] if l else Aijkl
    # Return the lth element of Aijkl only if l is specified and not None
    # else return the entire Aijkl array

# Question 7
# Finding Aijk from the individual terms calculated in the above formula

Aijk = np.sum(calc(),axis=1)

# Question 8
## (Ay(delx,0,z)-Ax(0,dely,z)-Ay(-delx,0,z)+Ax(0,-dely,z))/(4delx*dely)
## delx = 1

Ax = Aijk[0]
Ay = Aijk[1]

B = (Ay[2,1,:] - Ax[1,2,:] - Ay[0,1,:] + Ax[1,0,:])/4

# Question 9
## Plotting Bz(z) in a log-log plot

plt.figure(2)
plt.loglog(z,abs(B))
plt.xlabel(r"z-position (in cm) $\rightarrow$")
plt.ylabel(r"Magnitude of Bz $\rightarrow$")
plt.title(r"Log-Log plot of $|B_z(z)|$")
plt.savefig('images/fig2.png',dpi=100)
plt.show()

# Question 10
## Fitting the obtained values to B = cz^b

p = np.c_[np.ones(len(z)),np.log(z)]
log_c,b = np.linalg.lstsq(p,np.log(np.abs(B)),rcond=None)[0]
c = np.exp(log_c)
B_fit = c*(z**b)

plt.figure(3)
plt.loglog(z,np.abs(B),label="calculated value")
plt.loglog(z,np.abs(B_fit),label="Least Squares Fit")
plt.xlabel(r"z-position (in cm) $\rightarrow$")
plt.ylabel(r"Magnitude of $B_z \rightarrow$")
plt.title(r"Log-Log plot of $|B_z(z)|$")
plt.legend()
plt.savefig('images/fig3.png',dpi=100)
plt.show()

print(b,c)

log_c1,b1 = np.linalg.lstsq(p[100:],np.log(np.abs(B))[100:],rcond=None)[0]
c1 = np.exp(log_c1)
print(b1,c1)

log_c2,b2 = np.linalg.lstsq(p[250:],np.log(np.abs(B))[250:],rcond=None)[0]
c2 = np.exp(log_c2)
print(b2,c2)