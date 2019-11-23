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
K_Values=np.load('Plot_U/U_obs.npy')
K_Ucal=np.load('Plot_U/U_cal.npy')

V = FunctionSpace(mesh, 'P',1)
K=Function(V)
K_cal=Function(V)

K.vector()[:]=K_Values
K_cal.vector()[:]= K_Ucal
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
Z2=K_cal.vector().array()
#Z1=np.asarray(Z1)

print np.shape(x),np.shape(y),np.shape(Z1),

#np.save('/home/joaopaulo/workspacec/test1/src/default/adaptive2/Transient3/Var_FieldAnalytical2/VariableFieldVx2/Sol_z1exact001',Z1)
aux = np.min(Z2)
arquivo = open("h_"+str(valork)+".txt",'w')
arquivo.write("h_min: %.8f \n" %(aux))
arquivo.close() 
''
#---------------------------------------------Superfície 3D
'''
folga = 0.1
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
ax.plot_trisurf(x, y, Z2,cmap=cm.jet, linewidth=0.2)
ax.view_init(13,45)
ax.set_xlim(np.min(xy[:,0]),np.max(xy[:,0]))
ax.set_ylim(np.min(xy[:,1]),np.max(xy[:,1]))
ax.set_zlim(np.min(Z2),np.max(Z2)+folga)
t1=ax.set_xlabel(r'$X$',fontsize=16,color='black')
ax.set_ylabel(r'$Y$',rotation=0,fontsize=16,color='black')
ax.set_zlabel(r"$Carga$",rotation=0,fontsize=20,color='black')
plt.tricontour(xy[:,0],xy[:,1],Z2,20,cmap=cm.jet,linewidth=1,zdir='z',offset=np.min(Z2))
#perfis

DisCorte = 100
tol = 0.001  # avoid hitting points outside the domai
xx = np.linspace(np.min(xy[:,0]) + tol, np.max(xy[:,0]) - tol, 101)
#corteX = [2100.0, 2400.0, 3500.0 ] #locais de corte
corteX = range(2100, int(np.max(xy[:,0])), DisCorte)#locais de corte
corteY = range(2100, int(np.max(xy[:,1])), DisCorte)#locais de corte

for i in range(len(corteY)):
    points = [(x_, corteY[i]) for x_ in xx]  # Modificar local do corte
    w_line = np.array([K(point) for point in points])
    plt.plot(xx, w_line, '--', color='blue',zdir='y')
#perfis
yy = np.linspace(np.min(xy[:,1]) + tol, np.max(xy[:,1]) - tol, 101)
for i in range(len(corteX)):
    points = [(corteX[i], y_) for y_ in yy]  # Modificar local do corte
    w_line = np.array([K(point) for point in points])
    plt.plot(yy, w_line, '--', color='blue',zdir='x')

plt.legend(['Cortes a cada '+ str(DisCorte)+' m'], loc=3)
#plt.savefig('Plot_U/Condutividade.png')
'''
#---------------------------------------------curvas de nível
'''
fig = plt.figure()
plt.gca().set_aspect('equal')
plt.tricontour(xy[:,0],xy[:,1],Z1,31,cmap=cm.gist_heat_r,linewidth=1,zdir='z',offset=0.0)
plt.colorbar()
plt.tricontour(xy[:,0],xy[:,1],Z2,31,cmap=cm.jet,linewidth=1,zdir='z',offset=0.0)
plt.colorbar()
plt.xlim(np.min(xy[:,0]),np.max(xy[:,0]))
plt.ylim(np.min(xy[:,1]),np.max(xy[:,1]))
plt.plot(pcoorx_v,pcoory_v,'.',marker='o',markersize=8,color='m')
plt.plot(pcoorx_obs,pcoory_obs,'.',marker='o',markersize=8,color='r')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
#plt.savefig('Plot_U/curvas_nivel.png')
plt.show()
'''




#---------------------------------------------cortes
'''
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

folga = 0.3
fig, ax = plt.subplots()
tol = 0.001  # avoid hitting points outside the domai
x = np.linspace(np.min(xy[:,0]) + tol, np.max(xy[:,0]) - tol, 101)
points = [(x_, 2100.0) for x_ in x]  # Modificar local do corte
w_line = np.array([K(point) for point in points])
p_line = np.array([K_cal(point) for point in points])
plt.plot(x, w_line, '--', color='b')  # magnify w
plt.plot(x, p_line, '-', color='m')
plt.xlim(np.min(xy[:,0]),np.max(xy[:,0]))
plt.ylim((min(np.min(w_line),np.min(p_line)))-folga,(max(np.max(w_line),np.max(p_line)))+folga)
plt.grid(True)
plt.xlabel('$X$')
plt.ylabel('$Cargas$')
plt.legend(['Cargas Observadas', 'Cargas Calculadas'], loc=1)

#zoom 1
y1=w_line
y2 = p_line
zoom = 200
axins = zoomed_inset_axes(ax, zoom, loc=5)
axins.plot(x, y1, '--', color='b')
axins.plot(x, y2, '-',color='m')
axins.set_xlim(2100-1.5,2100+1.5)
axins.set_ylim((min(np.min(w_line),np.min(p_line)))-0.000042, min(np.min(w_line),np.min(p_line))+0.0007)
plt.xticks(visible = False)
#plt.xticks(visible = True)
plt.yticks(visible = True)
plt.title(str(zoom)+'x', loc = 'right')
plt.grid(True)
mark_inset(ax, axins, loc1=3, loc2=2, fc="none", ec="0.5")

#zoom 2
y1=w_line
y2 = p_line
zoom = 150
axins = zoomed_inset_axes(ax, zoom, loc=6)
axins.plot(x, y1, '--', color='b')
axins.plot(x, y2, '-',color='m')
axins.yaxis.set_ticks_position('right')
axins.set_xlim(1000-1.7,1000+1.7)
axins.set_ylim(9.9574107884426493-0.0005,9.9574107884426493+0.0005)
plt.xticks(visible = False)
plt.yticks(visible = True)
plt.title(str(zoom)+'x', loc = 'right')
plt.grid(True)
mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")

#zoom 3
y1=w_line
y2 = p_line
zoom = 100
axins = zoomed_inset_axes(ax, zoom, loc=3)
axins.plot(x, y1, '--', color='b')
axins.plot(x, y2, '-',color='m')
axins.yaxis.set_ticks_position('right')
axins.set_xlim(2016-2.5,2016+2.5)
axins.set_ylim(9.8583143524330143-0.0007,9.8583143524330143+0.0007)
plt.xticks(visible = False)
plt.yticks(visible = True)
plt.title(str(zoom)+'x', loc = 'right')
#plt.title(str(zoom)+'x')
plt.grid(True)
mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")

plt.draw()
plt.show()
'''





#plt.savefig('Plot_U/Corte.png')

