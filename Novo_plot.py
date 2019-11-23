# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 09:54:40 2019

@author: jj
"""
'''
from mpl_toolkits.mplot3d import Axes3D, axis3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt  
import matplotlib.tri as tri
import matplotlib as mpl
import numpy as np
from dolfin import *
parameters [ 'allow_extrapolation' ]  =  True
dolfin.parameters.reorder_dofs_serial = False #https://answers.launchpad.net/dolfin/+question/219220


mesh=Mesh("meshToWellAsVertex.xml")
xy=mesh.coordinates()
x=xy[:,0]
y=xy[:,0]
triangle=mesh.cells()

print triangle
print 'orientations', mesh.cell_orientations()

triang = tri.Triangulation(x, y,triangle)

X = x
Y = y

K_Ucal=np.load('Plot_U/U_cal.npy')
V = FunctionSpace(mesh, 'P',1)
K_cal=Function(V)
K_cal.vector()[:]= K_Ucal

Z=K_cal.vector().array()

minimoZ=min(Z); maximoZ=max(Z);
seq=np.linspace(minimoZ,maximoZ,50)

#--------------------------------
fig1=plt.figure(figsize=(10,8))

ax=fig1.gca(projection='3d')

ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
ax.plot_trisurf(X, Y,triangle, Z,cmap=cm.jet, linewidth=0.2)
surf=ax.plot_trisurf(X, Y, triangle, Z, linewidth=0.2, alpha=0.55, cmap=cm.gist_heat)

#surf=ax.tricontour(X,Y,triangle,Z,seq,triangle, zdir='Z',alpha=0.5,linewidth=0.50,cmap=cm.gist_heat_r, offfset=0)
ax.view_init(45,-57)
t1=ax.set_xlabel(r'$X$',fontsize=16,color='black')
ax.set_ylabel(r'$Y$',rotation=45,fontsize=16,color='black')
ax.set_zlim3d(minimoZ,maximoZ)
plt.tight_layout()
ax.set_zlabel(r"$Carga$",rotation=90,fontsize=20,color='black')

'''
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)

# Plot the 3D surface
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
'''
# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_zlim(-100, 100)
'''
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()