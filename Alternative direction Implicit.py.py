import numpy as np 
import scipy 
import scipy.linalg 
import scipy.sparse 
import scipy.sparse.linalg 
from math import sinh 

import time
st = time.time()

# Set temperature at the top,bottom,left,right 
Ttop=300 
Tbottom=50 
Tleft=75.0 
Tright=100.0 
xmax=2.4
ymax=3 
# Set simulation parameters 
#need hx=(1/nx)=hy=(1.5/ny) 
Nx = 20 
h=xmax/Nx 
Ny = int(ymax/h) 
nx = Nx-1 
ny = Ny-1 
n = (nx)*(ny) #number of unknowns 


d = np.ones(n) # diagonals 
b = np.zeros(n) #RHS 
d0 = d*-4 
d1 = d[0:-1] 
d5 = d[0:-ny] 
A = scipy.sparse.diags([d0, d1, d1, d5, d5], [0, 1, -1, ny, -ny], format='csc') 
#alternatively (scalar broadcasting version:) 
#A = scipy.sparse.diags([1, 1, -4, 1, 1], [-5, -1, 0, 1, 5], shape=(15, 15)).toarray() 
# set elements to zero in A matrix where BC are imposed 
for k in range(1,nx): 
    j = k*(ny) 
    i = j - 1 
    A[i, j], A[j, i] = 0, 0 
    b[i] = -Ttop 
b[-ny:]+=-Tright #set the last ny elements to -Tright 
b[-1]+=-Ttop #set the last element to -Ttop 
b[0:ny-1]+=-Tleft #set the first ny elements to -Tleft 
b[0::ny]+=-Tbottom #set every ny-th element to -Tbottom 

theta = scipy.sparse.linalg.spsolve(A,b) #theta=sc.linalg.solve_triangular(A,d) 
theta2=scipy.linalg.solve(A.toarray(),b) 
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
for j in range(1,ny+1): 
    for i in range(1, nx + 1): 
        T[j, i] = theta[j + (i-1)*ny - 1]

#ploting the 3D surface plot
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
#fig.colorbar(surf, shrink=0.5, aspect=5) 
print("Time of execution = "+str(time.time()-st))
plt.show() 