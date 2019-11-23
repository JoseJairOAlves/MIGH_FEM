'''
Created on 13/09/2019 @author: dell


'''

import matplotlib.pyplot  as plt
import numpy as np
from dolfin import *
parameters [ 'allow_extrapolation' ]  =  True
dolfin.parameters.reorder_dofs_serial = False #https://answers.launchpad.net/dolfin/+question/219220
###################################################################################################################


#LENDO A LISTA DE VALORES DO RMSEH E DO PHI
rmseh=np.load("Analise_erro/rmseh_list.npy")
phi=np.load("Analise_erro/phi_list.npy")

#fIGURA DA EVOLUCAO DO ALGORITMO PARA RMSEH E PHI. 
fig1=plt.figure()
p1,=plt.plot(rmseh,'.-')
plt.grid(True)
plt.xlabel('Ciclos')
plt.ylabel('RMSEH (metros)')
xs = np.arange(0,len(rmseh),1)
ys = rmseh
for x,y in zip(xs,ys):
    label = "{:.3f}".format(y)
    if x != 0:
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='left') # horizontal alignment can be left, right or center
plt.savefig('Analise_erro/RMSEH.png', transparent=True)#, transparent = True)
plt.show()

fig2=plt.figure()
p2,=plt.plot(phi,'.-')
plt.grid(True)
plt.xlabel('Ciclos')
plt.ylabel(r'$\phi$ (Graus)')
xs = np.arange(0,len(phi),1)
ys = phi
for x,y in zip(xs,ys):
    label = "{:.3f}".format(y)
    if x != 0:
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='left') # horizontal alignment can be left, right or center
plt.savefig('Analise_erro/angulo.png',transparent=True)#, transparent = True)
plt.show()


