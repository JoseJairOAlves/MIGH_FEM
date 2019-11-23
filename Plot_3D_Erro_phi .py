# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:31:49 2019

@author: jj
"""

###################################################################################################################
#PLOTTING 3D FIGURES 
#=======================================================================================================
# import to generate visualizations of finite element solutions
#import scitools.BoxField
#import scitools.easyviz as ev
# the preference is for matplotlib plotting
#==================================================================================================================

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt  
import matplotlib.tri as tri
import numpy as np
from dolfin import *
parameters [ 'allow_extrapolation' ]  =  True
dolfin.parameters.reorder_dofs_serial = False #https://answers.launchpad.net/dolfin/+question/219220
###################################################################################################################
#LENDO MALHA JA PRONTA
mesh=Mesh("meshToWellAsVertex.xml")
K_Values=np.load('Plot_phi/phi_plot.npy')

V = FunctionSpace(mesh, 'P',1)
K=Function(V)

K.vector()[:]=K_Values
#plot(K,mesh=mesh,title = 'Gradiente')

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
#Z1=np.asarray(Z1)

print np.shape(x),np.shape(y),np.shape(Z1),
'''
#np.save('/home/joaopaulo/workspacec/test1/src/default/adaptive2/Transient3/Var_FieldAnalytical2/VariableFieldVx2/Sol_z1exact001',Z1)
arquivo = open("phi_"+str(valork)+".txt",'w')
arquivo.write("Erro_min: %.8f \n" %(aux))
aux = np.max(K.vector().array())
arquivo.write("Erro_max: %.8f \n" %(aux))
aux = np.mean(K.vector().array())
arquivo.write("Erro_media: %.8f \n" %(aux))
aux = np.std(K.vector().array())
arquivo.write("Erro_des_p: %.8f \n" %(aux))
arquivo.close()

'''
#---------------------------------------------Superfície 3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
ax.plot_trisurf(x, y, Z1,cmap=cm.jet, linewidth=0.2)
ax.view_init(30,-120)
plt.xlim(np.min(xy[:,0]),np.max(xy[:,0]))
plt.ylim(np.min(xy[:,1]),np.max(xy[:,1]))
t1=ax.set_xlabel(r'$X$',fontsize=16,color='black')
ax.set_ylabel(r'$Y$',rotation=45,fontsize=16,color='black')
ax.set_zlabel(r"$graus$",rotation=90,fontsize=20,color='black')
#plt.savefig('Plot_phi/Condutividade.png')


#---------------------------------------------curvas de nível
'''
n_curvas = 30
fig = plt.figure()
plt.gca().set_aspect('equal')
plt.tricontour(xy[:,0],xy[:,1],Z1,n_curvas,cmap=cm.jet,linewidth=0.6,zdir='z',offset=0.0)#)gist_heat_r)# *args, **kwargs)
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(np.min(xy[:,0]),np.max(xy[:,0]))
plt.ylim(np.min(xy[:,1]),np.max(xy[:,1]))
plt.grid(True)
#plt.colorbar(Z1)
#plt.savefig('Plot_phi/test.png')
plt.show()
'''