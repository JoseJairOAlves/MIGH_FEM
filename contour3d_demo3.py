from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
import matplotlib.tri as tri
import numpy as np
from dolfin import *
parameters [ 'allow_extrapolation' ]  =  True
dolfin.parameters.reorder_dofs_serial = False #https://answers.launchpad.net/dolfin/+question/219220
###################################################################################################################
#LENDO MALHA JA PRONTA
mesh=Mesh("meshToWellAsVertex.xml")
K_Values=np.load('Plot_k/K_Function.npy')

V = FunctionSpace(mesh, 'P',1)
K=Function(V)

K.vector()[:]=K_Values
plot(K,mesh=mesh,title = 'Gradiente')

#PLOTTING 3D FIGURES 
# DEFINITION OF MESH ENTITIES TO PLOT WITH MATPLOTLIB

xy=mesh.coordinates()# the mesh coordinates
#print xy

x=xy[:,0]# the x coordinates
print len(x)
y=xy[:,1]# the y coordinates
triangle=mesh.cells()# a nx3 matrix storing triangles conectivities
triang = tri.Triangulation(x, y,triangle)
Z1=K.vector().array()
Z1=np.asarray(Z1)


#---------------------------------------------Superfície 3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
ax.plot_trisurf(x, y, Z1,cmap=cm.jet, linewidth=0.2)
ax.view_init(47,-57)
ax.set_xlim(np.min(xy[:,0]),np.max(xy[:,0]))
ax.set_ylim(np.min(xy[:,1]),np.max(xy[:,1]))
t1=ax.set_xlabel(r'$X$',fontsize=16,color='black')
ax.set_ylabel(r'$Y$',rotation=45,fontsize=16,color='black')
ax.set_zlabel(r"$K$",rotation=0,fontsize=20,color='black')

''''
fig = plt.figure()
ax = fig.gca(projection='3d')
#X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
#cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

plt.show()
