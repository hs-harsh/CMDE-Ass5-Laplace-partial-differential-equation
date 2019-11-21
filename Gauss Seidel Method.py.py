import time 
from math import sinh 
from astropy.units import dT
import numpy as np

import time
st = time.time()


Ttop=300
Tbottom=50.0 
Tleft=75.0 
Tright=100.0 

xmax=2.4 
ymax=3
# Set simulation parameters #need hx=(1/nx)=hy=(1.5/ny) 
Nx = 20 
h=xmax/Nx 
Ny = int(ymax/h)

nx = Nx-1 
ny = Ny-1 
n = (nx)*(ny) #number of unknowns 
# surfaceplot: 
x = np.linspace(0, xmax, Nx + 1) 
y = np.linspace(0, ymax, Ny + 1) 
X, Y = np.meshgrid(x, y) 
T = np.zeros_like(X)

# set the imposed boudary values 
T[-1,:] = Ttop 
T[0,:] = Tbottom 
T[:,0] = Tleft 
T[:,-1] = Tright 
omega = 1.5 
reltol=1.0e-3 
iteration = 0 
rel_res=1.0 
# Gauss-Seidel iterative solution 
while (rel_res > reltol): 
    dTmax=0.0 
    for j in range(1,ny+1): 
        for i in range(1, nx + 1): 
            R = (T[j,i-1]+T[j-1,i]+T[j,i+1]+T[j+1,i]-4.0*T[j,i]) 
            dT = 0.25*omega*R 
            T[j,i]+=dT
            dTmax=np.max([np.abs(dT),dTmax])
    rel_res=dTmax/np.max(np.abs(T))
    iteration+=1   
print ("Gauss-Seidel \t Nr. iterations "+str(iteration))

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter 
fig = plt.figure() 
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, T, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False) 
ax.set_zlim(0, Ttop+10) 
ax.set_xlabel('x') 
ax.set_ylabel('y') 
ax.set_zlabel('T [$^o$C]')
nx=4 
xticks=np.linspace(0.0,xmax,nx+1) 
ax.set_xticks(xticks) 

ny=4 
yticks=np.linspace(0.0,ymax,ny+1) 
ax.set_yticks(yticks) 
nTicks=5 
dT=int(Ttop/nTicks) 
Tticklist=range(0,Ttop+1,dT) 
ax.set_zticks(Tticklist)
print("Time of execution = "+str(time.time()-st))
plt.show() 