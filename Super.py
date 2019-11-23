'''Created on 13/09/2019 @author: dell'''

import matplotlib.pyplot  as plt
import numpy as np

from dolfin import *
parameters [ 'allow_extrapolation' ]  =  True
dolfin.parameters.reorder_dofs_serial = False #https://answers.launchpad.net/dolfin/+question/219220
###################################################################################################################
#LENDO MALHA JA PRONTA
mesh=Mesh("meshToWellAsVertex.xml")

V = FunctionSpace(mesh, 'P',1); 
u_obsArray=np.load('Plot_k/K_Function.npy');
u_obs=Function(V)
u_obs.vector()[:]=u_obsArray


#PLOTANDO OS VALORES
#plot(rmseh_Function,mesh=mesh,interactive=True)
#plot(T_f,mesh=mesh,interactive=True)

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D,axis3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib as plt
import numpy as np



from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt 


# DEFINITION OF MESH ENTITIES TO PLOT WITH MATPLOTLIB
xy=mesh.coordinates()# the mesh coordinates
x=xy[:,0]# the x coordinates
y=xy[:,1]# the y coordinates
triangle=mesh.cells()# a nx3 matrix storing triangles conectivities
# Now create the Triangulation.
# (Creating a Triangulation without specifying the triangles results in the
# Delaunay triangulation of the points.)
triang = tri.Triangulation(x, y,triangle)

X=x#u_box.grid.coorv[X];
Y=y#u_box.grid.coorv[Y];
Zobs=u_obs.vector().array()#u_box.values

#Z[x>0]=np.NAN
#Ztheta=np.arctan2((y-600),(x-600.0))
#Psi=Psi.vector().array()
seq= np.linspace(np.min(Zobs),np.max(Zobs),30)

#DADOS DA MALHA
x0=0.0; y0=0.0; x1=4200.0; y1=4200.0;


tol = 0.001  # avoid hitting points outside the domai
x = np.linspace(np.min(xy[:,0]) + tol, np.max(xy[:,0]) - tol, 101)
points = [(x_, 3400.0) for x_ in x]  # Modificar local do corte
w_line = np.array([u_obs(point) for point in points])



#------------------------------------------------------------------------------------------------
fig2 = plt.figure(figsize=(8,6))
#plt.figure(num=None, figsize=(8, 6), dpi=380, facecolor='w', edgecolor='k')
ax = fig2.gca(projection='3d')
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
#ax.plot_trisurf(X,Y,Z,linewidth=0.30,alpha=0.3,cmap=cm.gist_heat)# rstride=1, cstride=2,
cont1 = ax.tricontour(X,Y,Zobs,20,zdir='z',linewidth=0.50,alpha=0.73,
                     cmap=cm.gist_heat_r,antialiased=True,offset=10.1)# rstride=1, cstride=2,
plt.plot(x, w_line, '--', color='b',zdir='y')
'''
cont2 = ax.tricontour(X,Y,Zobs,20,zdir='y',linewidth=0.50,alpha=0.73,
                     cmap=cm.gist_heat_r,antialiased=True,offset=y1+200)# rstride=1, cstride=2,
cont3 = ax.tricontour(X,Y,Zobs,20,zdir='x',linewidth=0.50,alpha=0.73,
                     cmap=cm.gist_heat_r,antialiased=True,offset=x1+200)# rstride=1, cstride=2,
'''
surf = ax.plot_trisurf(X,Y,Zobs,linewidth=0.20,alpha=0.73,cmap=cm.jet)#,rstride=1)
ax.view_init(47,-57)
#ax.set_xlim(x0,x1)
#ax.set_ylim(y0,y1) 
ax.set_xlabel('x',fontsize=20,color='black')
ax.set_ylabel('y',rotation=90,fontsize=20,color='black')
#ax.text(1000.0,2500.0,0.80,r"$C=C(x,y,t=T)$",fontsize=20,color='blue')
#ax.set_zlim3d(10.0, 25.) 
#ax.set_zlabel("Erro", rotation=90,fontsize=20,color='black')
#plt.tight_layout()
#plt.savefig('AEM08_fig01.png')
plt.show()


