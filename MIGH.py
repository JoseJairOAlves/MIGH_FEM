# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:43:12 2019

@author: Jair & Renato
Atualizacoes: João Paulo
"""

###################################################################################################################
#BIBLIOTECA

import numpy as np
from dolfin import *
from mshr import *

parameters [ 'allow_extrapolation' ]  =  True
dolfin.parameters.reorder_dofs_serial = False 

###################################################################################################################
#FUNÇÕES
#-----------------------------------------------Ajuste da Malha
def MeshPassingToWellAsVertex(mesh,coorx,coory):
    ''' author: jpmdossantos@yahoo.com.br
    Based on code from Fenics website and Q&A'''
    #Example From: https://fenicsproject.org/qa/9763/method-compute_closest_point
    coords=mesh.coordinates()
    print np.shape(coords)
    
    tree = mesh.bounding_box_tree()
    
    point_cloud = [dolfin.Point(point) for point in mesh.coordinates()]
    tree.build(point_cloud, 2)
    p=Point(coorx[0],coory[0])
    #print p
    p_i, distance = tree.compute_closest_point(p)
    print p_i
    
    lengthCoor=np.shape(coords)[0]
    print'lenght coordinates matrix Xdirection', lengthCoor
    for i in range(len(coorx)):
        print i
        p=Point(coorx[i],coory[i])
        #tree = mesh.bounding_box_tree()
        #point_cloud = [dolfin.Point(point) for point in mesh.coordinates()]
        #tree.build(point_cloud, 2)
        p_i, distance = tree.compute_closest_point(p)   
        print "p:", p.str()
        print "closest coordinate point by reference to index",coords[p_i]
        print "closest point:",p_i,point_cloud[p_i].str()
        
        #os indices por referencia sao aqueles da matriz numpy. Logo basta trocar direto
        coords[p_i][0]=coorx[i];coords[p_i][1]=coory[i];
          
        #XMesh= coords[vertex.index()][0];YMesh= coords[vertex.index()][1]
    return mesh
    
#-----------------------------------------------Refinamento da Malha
def FunctionToRefine(mesh,listaCoorX,listaCoorY,lista):
    """
    ''' author: jpmdossantos@yahoo.com.br
    Based on code from Fenics website and Q&A''' 
    Function to refine a given mesh around Wells with Coordinates (X,Y) given as 
    listaCoorX and listaCoorY. Loop of refinement extends for the size of lista and lista 
    contains the distance for decision criteria for refinement
    Usage:
    mesh=FunctionToRefine(mesh,listaCoorX,listaCoorY,lista)
    
    """
    print "Mesh before refinement %s"%str(mesh)
    for i in range(len(lista)):
        print("marking for refinement at iteraction ",i)
        
        # Mark cells for refinement
        #cell_markers = CellFunction("bool", mesh)
        cell_markers = MeshFunction("bool",mesh, mesh.topology().dim())
        cell_markers.set_all(False)
        
        for c in cells(mesh):
            for w in range(len(listaCoorX)):#shapeWells[0]):    
                #print listaCoorX[w],listaCoorY[w] 
                p=Point(listaCoorX[w],listaCoorY[w])
                #print listaCoorX[w],listaCoorY[w]
                if (c.midpoint().distance(p) < lista[i]):
                    cell_markers[c] = True
                #else:
                #    cell_markers[c] = False
        mesh = refine(mesh, cell_markers)
        print "%s"%str(mesh)
        #plot(mesh)
        #mesh_points = mesh.coordinates()
        #mesh_tris = mesh.cells()#np.array(mesh.elements)
        #import matplotlib.pyplot as pt
        #pt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
        #pt.plot(listaCoorX,listaCoorY,'.',marker='o',markersize=12,color='m')
        #pt.axis([-.1,1.1,-.1,1.1])
        #pt.xlabel('x');pt.ylabel('y')
        #pt.show()
    return mesh

#-----------------------------------------------Condições de contorno pontuais
def CreateFunctionPointwise(hp_obs):
    arquivo = open("FunctionPointwise.py",'w')
    arquivo.write("from fenics import *\n")
    arquivo.write("def FunctionPointwise(V, bc_obs, pcoorx_obs, pcoory_obs, hp_obs):\n")
    arquivo.write("     tol = DOLFIN_EPS\n")
    arquivo.write("\n")
    
    for i in range( len(hp_obs)):
        arquivo.write("     def condicao_p%i (x): \n" %(i))
        arquivo.write("         return near(x[0], pcoorx_obs[%i],tol) and near(x[1], pcoory_obs[%i],tol)\n" %(i,i))
        arquivo.write("     bcp_%i = DirichletBC(V, hp_obs[%i], condicao_p%i, method='pointwise')\n" %(i,i,i))
        arquivo.write("     bc_obs.append(bcp_%i)\n" %(i))
        arquivo.write("\n")
        
    arquivo.write("     return bc_obs")
    arquivo.close()    
    return 

####################################################################################################################
#CONFIGURAÇÕES/ENTRADAS

#-----------------------------------------------Cordenadas e Altura dos Poços de Observação

pcoorx_obs = [1800.0, 3600.0, 600.0, 2200.0, 800.0, 3600.0]
pcoory_obs = [1600.0, 3000.0, 3400.0, 2800.0, 800.0, 800.0]
hp_obs = [9.9135966045882817, 9.9822728930903981, 9.9880481825748433, 9.9259346111578957, 9.9839878519736711, 9.9880508018551044]
#hp_obs = [20, 20, 20, 20, 20, 20]
#-----------------------------------------------Vazões de Coordenadas dos Poços de Bombeamento
pcoorx_v = [2100.0]
pcoory_v = [2100.0]
vazao = [-2000.0]#m3/s

#-----------------------------------------------Variáveis Hidrogeológicas Iniciais
valork=5*pow(10,9) #Condutividade Inicial
b=10.0 #Espessura do aquífero

#Valores que Garantam a Plausabilidade
k_min = 1#1E-7 #Condutividade Mínima
k_max = 5*pow(10,9)#1E-2 #Condutividade Máxima

#-----------------------------------------------Critérios de Pardas do MIGH
inter = 10 #Número Máximo de Interação
RMSEH_MAX = 1.0E-15 #Número Erro RMSEH Mínimo

#-----------------------------------------------Dados da malha
#dados inicias
resolution = 130 #resolução
x0=0.0 #fronteira esquerda
x1=4200.0 #fronteira direita

#refinamento do poço
lista=[100.0, 50.0, 25.0, 10.0, 1.0, 0.5, 0.1]

#malha Retangular

y0=0.0
y1=4200.0
nx = 50;ny = 30;
P0=Point(x0,y0); P1=Point(x1,y1);
rect = Rectangle(P0,P1)
mesh=generate_mesh(rect,resolution,"cgal")

###################################################################################################################
#REFINAMENTO DA MALHA
size_max = max(len(pcoorx_obs),len(pcoorx_v))
size_min = min(len(pcoorx_obs),len(pcoorx_v))

#-----------------------------------------------Ordenando os vetores
if len(pcoorx_obs)==size_max:
    LXmax = pcoorx_obs
    LYmax = pcoory_obs
    LXmin = pcoorx_v
    LYmin = pcoory_v
else:
    LXmax = pcoorx_v
    LYmax = pcoory_v
    LXmin = pcoorx_obs
    LYmin = pcoory_obs
    
#-----------------------------------------------Inserindo Valores Diferentes
for i in range(size_min):
    soma = 0
    for j in range(size_max):
        if LXmax[j]==LXmin[i] and LYmax[j]==LYmin[i]:
            soma += 1
            break
    if soma == 0:
        LXmax.append(LXmin[i])
        LYmax.append(LYmin[i])

#-----------------------------------------------Refinando a Malha 
#mesh = FunctionToRefine(mesh, LXmax, LYmax, lista)
#mesh = MeshPassingToWellAsVertex(mesh, LXmax, LYmax)
mesh = MeshPassingToWellAsVertex(mesh, pcoorx_obs, pcoory_obs)
mesh = MeshPassingToWellAsVertex(mesh, pcoorx_v, pcoory_v)
coor = mesh.coordinates()

#-----------------------------------------------Plot da Malha 
'''
mesh_tris = mesh.cells()#np.array(mesh.elements)
import matplotlib.pyplot as pt
pt.triplot(coor[:, 0], coor[:, 1], mesh_tris, color='b')
pt.plot(pcoorx_v,pcoory_v,'.',marker='o',markersize=8,color='m')
pt.plot(pcoorx_obs,pcoory_obs,'.',marker='o',markersize=8,color='r')
pt.xlabel('x');pt.ylabel('y')
pt.gca().set_aspect('equal')
pt.xlim(np.min(coor[:,0]),np.max(coor[:,0]))
pt.ylim(np.min(coor[:,1]),np.max(coor[:,1]))
#plt.legend(['Cortes a cada ', 'Cortes a cada '], loc=3)
#pt.savefig("MeshToVertex01After")
pt.show()
'''
File("meshToWellAsVertex.xml")<<mesh
mesh=Mesh("meshToWellAsVertex.xml")
#plot(mesh)

###################################################################################################################
#FUNÇÕES DE ESPAÇOS 
V = FunctionSpace(mesh, 'P',1)
V_vec = VectorFunctionSpace(mesh, "CG", 1)

###################################################################################################################
#CONDIÇÕES DE CONTORNO

#-----------------------------------------------Condição de Contorno (Carga) Oeste
u_W=Constant(10.0) 
def u0w_boundary(x, on_boundary):
     #tol = 1E-6#DOLFIN_EPS
     return on_boundary #and near(x[0],x0,tol)
bcw = DirichletBC(V, u_W, u0w_boundary)

#-----------------------------------------------Condição de Contorno (Carga) Leste
#u_E=Constant(10.0) 
#def u0e_boundary(x, on_boundary):
#     tol = 1E-6#DOLFIN_EPS
#     return on_boundary and near(x[0],x1,tol) 
#bce = DirichletBC(V, u_E, u0e_boundary)

bc_cal = [bcw]#,bce]
bc_obs = [bcw]#,bce]

#-----------------------------------------------Condição de Contorno Pontuais

CreateFunctionPointwise(hp_obs)#criando funçao dos PointWise
from FunctionPointwise import * #importando funçao
bc_obs = FunctionPointwise(V, bc_obs, pcoorx_obs, pcoory_obs, hp_obs)

###################################################################################################################
#CLASSE DE K 
class k01(Expression):
        def __init__(self,mesh,valork):
            self.mesh    = mesh
            #self._ufl_element = element
        def eval_cell(self, values, x, ufc_cell):
            #dolfin_cell = Cell(self.mesh, ufc_cell.index)
            values[0]=valork#(vx(x[0],x[1])*dolfin_cell.diameter())/(dxx(x[0],x[1]))   
        #def value_shape(self):
        #   return (1,)         
#k01= k01(element=V.ufl_element()); 
#f = interpolate(f,V)
k0=k01(mesh,valork)
k0=interpolate(k0,V)
#plot(k0, True, title = 'Condutividade inicial')

T=Function(V);T.vector()[:]=b*k0.vector().array()#Expression('k0*b',b=b,k0=k0)#*24*60*60*b# m2/dia
T=interpolate(T,V)
#plot(T, title = 'Transmissividade inicial')

###################################################################################################################
#MÉTODO INTERATIVO DO GRADIENTE HIDRAULICO

#listas para capturar os valores rmseh e phi
RMSEH_list=[];phi_list=[]
ciclo = 0
tolerance=1.0#criterio de parada
#-------------------------
arquivo = open("Erros_"+str(valork)+".txt",'w')
arquivo.write("RMSEH; PHI \n")
#---------------------------------------
while(tolerance>RMSEH_MAX):#(ciclo<inter):#(ciclo < inter or epsilon<1.0E-06):
    print("ciclo %s"%ciclo)
    #----------------------------------------------------------------
    #cargas observadas
    u_obs = TrialFunction(V)
    v_obs = TestFunction(V)
    a_obs = inner(T*nabla_grad(u_obs), nabla_grad(v_obs))*dx
    L_obs = Constant(0.0)*v_obs*dx
    A_obs = assemble(a_obs)
    rhs1_obs = None
    rhs_obs = assemble(L_obs, tensor=rhs1_obs)
    u_obs = Function(V)
    for bc in bc_obs:
        bc.apply(A_obs,rhs_obs)
    for i in range (len(vazao)):
        delta_obs = PointSource(V, Point(pcoorx_v[i], pcoory_v[i]), vazao[i])
        delta_obs.apply(rhs_obs)
    solve(A_obs, u_obs.vector(), rhs_obs)
    #plot(u_obs, title = 'Cargas observadas')
    
    grad_obs = project(-1*nabla_grad(u_obs),V_vec)
    grad_obsx, grad_obsy = grad_obs.split(deepcopy = True)
    grad_obsx = grad_obsx.vector().array()
    grad_obsy = grad_obsy.vector().array()
    #plot(grad_obs, title = 'Gradiente observado')
    
    #----------------------------------------------------------------
    #cargas calculadas
    u_cal = TrialFunction(V)
    v_cal = TestFunction(V)
    a = inner(T*nabla_grad(u_cal), nabla_grad(v_cal))*dx
    L = Constant(0.0)*v_cal*dx
    A, rhs = assemble_system(a, L, bc_cal)
    u_cal = Function(V)
    for i in range (len(vazao)):
        delta_obs = PointSource(V, Point(pcoorx_v[i], pcoory_v[i]), vazao[i])
        delta_obs.apply(rhs)
    solve(A, u_cal.vector(), rhs)
    #plot(u_cal, title = 'Cargas calculdas')
    
    grad_cal = project(-1*nabla_grad(u_cal),V_vec)
    grad_calx, grad_caly = grad_cal.split(deepcopy = True)
    grad_calx = grad_calx.vector().array()
    grad_caly = grad_caly.vector().array()
    #plot(grad_cal, title = 'Gradiente calculado')

    h_obs = u_obs.vector().array()
    h_cal = u_cal.vector().array()
                
    #Cálculo dos erros
    #Vou usar para plotar RMSEH2 por isso nao estou redefinindo.
    RMSEH1 = h_obs - h_cal
    RMSEH2 = pow(RMSEH1,2)
    RMSEH = np.sum(RMSEH2)
    RMSEH2 = np.sqrt(RMSEH2)
    RMSEH = RMSEH/len(h_obs)
    RMSEH = np.sqrt(RMSEH)
    RMSEH_list.append(RMSEH)
    
    
    aux1 = grad_obsx*grad_calx + grad_obsy*grad_caly;
    _grad_obs = np.sqrt(grad_obsx*grad_obsx + grad_obsy*grad_obsy)
    _grad_cal = np.sqrt(grad_calx*grad_calx + grad_caly*grad_caly)
    

    aux1=aux1/(_grad_obs*_grad_cal)
    values=aux1[aux1>1.0]
    #print('valores do argumento do acos maiores que 1',values)
    #''' solucao do problema de phi<-1.  ou phi>1.'''
    aux1[aux1>1.0]=1.0
    aux1[aux1<-1.0]=-1.0
    phi = np.arccos(aux1)
    phi_plot = phi 
    #print('valores min e max de phi',np.min(phi),np.max(phi))
    phiSum = np.sum(phi)/len(h_obs)
    phi_list.append(phiSum)
    
#-------------------------    
    arquivo.write("%.8f; %.8f \n" %(RMSEH, phiSum))
#-------------------------
    #tolerance=RMSEH
    #print 'VALOR DO RMSEH:%8g  VALOR DO PHI: %8g' % (RMSEH, phiSum)

    Tarray=T.vector().array()
    k0Array=k0.vector().array()
    
    #for i in range(len(pcoorx_obs)):
    #    print('(%i,%i)'%(pcoorx_obs[i],pcoory_obs[i]),(u_cal(pcoorx_obs[i],pcoory_obs[i])))
    
    for i in range(len(Tarray)):
        valor = k0Array[i]*(_grad_cal[i]/_grad_obs[i])
        if valor > k_max:
            k0Array[i] = k_max
        elif valor < k_min:
            k0Array[i] = k_min
        else:
            k0Array[i] = valor
        Tarray[i]=k0Array[i]*b#.append(k_n[i]*b)#h_obs[i])#ver com Paulo
        
    k0=Function(V);k0.vector()[:]=k0Array
    T=Function(V);T.vector()[:]=Tarray
    #plot(T,title = 'Transmissividade', interactive = True )
    
    if ciclo>=inter:
        print("MAXIMO NUMERO DE ITERACOES ATINGIDO %d"%ciclo)
        break
    
    ciclo = ciclo + 1

###################################################################################################################
#SAVE PARA GRÁFICOS
arquivo.close() 
    
np.save('Analise_erro/rmseh_list.npy',RMSEH_list)
np.save('Analise_erro/phi_list.npy',phi_list)
np.save('Plot_k/K_Function.npy',T.vector().array()/b)

np.save('Erro_h/rmseh_function.npy',RMSEH2)
np.save('Plot_phi/phi_plot.npy',phi_plot*180.0/pi)

np.save('Plot_U/U_obs.npy',u_obs.vector().array())
np.save('Plot_U/U_cal.npy',u_cal.vector().array())
