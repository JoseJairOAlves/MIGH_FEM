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
K_Values=np.load('Plot_k/K_Function.npy')

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

#np.save('/home/joaopaulo/workspacec/test1/src/default/adaptive2/Transient3/Var_FieldAnalytical2/VariableFieldVx2/Sol_z1exact001',Z1)

arquivo = open("K"+str(valork)+".txt",'w')
aux = np.mean(K.vector().array())
arquivo.write("K_medio: %.8f \n" %(aux))
aux = np.std(K.vector().array())
arquivo.write("K_des: %.8f \n" %(aux))
arquivo.close()

#---------------------------------------------Superfície 3D
'''
fig = plt.figure()
plt.gca().set_aspect('equal')
ax = fig.gca(projection='3d')
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
ax.plot_trisurf(x, y, Z1,cmap=cm.jet, linewidth=0.2)
ax.view_init(30,-120)
ax.set_xlim(np.min(xy[:,0]),np.max(xy[:,0]))
ax.set_ylim(np.min(xy[:,1]),np.max(xy[:,1]))
t1=ax.set_xlabel(r'$X$',fontsize=16,color='black')
ax.set_ylabel(r'$Y$',rotation=0,fontsize=16,color='black')
ax.set_zlabel(r"$K$",rotation=0,fontsize=20,color='black')
'''
'''
#perfis
tol = 0.001  # avoid hitting points outside the domai

xx = np.linspace(np.min(xy[:,0]) + tol, np.max(xy[:,0]) - tol, 101)
for i in range(len(LYmax)):
    points = [(x_, LYmax[i]) for x_ in xx]  # Modificar local do corte
    w_line = np.array([K(point) for point in points])
    plt.plot(xx, w_line, '--', color='black',zdir='y')

#perfis
yy = np.linspace(np.min(xy[:,1]) + tol, np.max(xy[:,1]) - tol, 101)
for i in range(len(LXmax)):
    points = [(LXmax[i], y_) for y_ in yy]  # Modificar local do corte
    w_line = np.array([K(point) for point in points])
    plt.plot(yy, w_line, '--', color='black',zdir='x')

#plt.savefig('Plot_k/Condutividade.png')

'''

#---------------------------------------------curvas de nível
#plt.gca().set_aspect('equal')
######plt.tricontour(xy[:,0],xy[:,1],Z1,50,cmap=cm.jet,linewidth=0.01,zdir='z',offset=0.0)#)gist_heat_r)# *args, **kwargs)
#plt.colorbar()
#plt.title('Contour plot of user-specified triangulation')
'''
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(np.min(xy[:,0]),np.max(xy[:,0]))
plt.ylim(np.min(xy[:,1]),np.max(xy[:,1]))
'''
#plt.colorbar()
#plt.grid(True)
#plt.gca().set_aspect('equal')
#plt.colorbar(Z1)
#plt.savefig('Plot_k/test.png')
#####plt.show()

#---------------------------------------------Histograma
'''
x = K.vector().array()
arquivo = open("Histograma_"+str(valork)+".txt",'w')
for i in range(len(x)):
    arquivo.write("%.6f \n" %(x[i]))
arquivo.close()    
 '''

'''
fig, ax = plt.subplots()
test = int(sqrt(len(x)))
H = ax.hist(x, bins=range(100, 1100, 75), alpha=1, histtype='bar')
#H = ax.hist(x, bins=test, alpha=1, histtype='bar')
plt.xlabel('Condutividade')
plt.ylabel("Frequencia")
#ax.set_ylim(np.min(x),np.max(x))
#pt.gca().set_aspect('equal')
#plt.text(100, 1000, r'$\mu = 1000, \ \sigma=15$')
'''
#---------------------------------------------Showing Images
'''
I = np.random.random((100, 100))
I += np.sin(np.linspace(0, np.pi, 100))
fig, ax = plt.subplots()
im = ax.imshow(I, cmap=plt.cm.jet)
fig.colorbar(im, ax=ax, orientation='horizontal')
'''

#---------------------------------------------Showing Images
'''
fig = plt.figure()
MIN = K.vector().array()
MIN = np.min(MIN)
fig, ax = plt.subplots()
plt.gca().set_aspect('equal')
im = ax.hexbin(x, y, gridsize=10, vmax=0.5)
fig.colorbar(im, ax=ax)
'''
'''
#fig = plt.figure()
fig, ax = plt.subplots()
plt.gca().set_aspect('equal')
H = ax.hist2d(x, y, bins=10)
fig.colorbar(H[3], ax=ax)
'''
#---------------------------------------------Showing Images
'''
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(x, y, Z1, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(x, y, Z1, zdir='z', offset=0.01, cmap=cm.coolwarm)
cset = ax.contour(x, y, Z1, zdir='x', offset=40, cmap=cm.coolwarm)
cset = ax.contour(x, y, Z1, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

plt.show()
'''