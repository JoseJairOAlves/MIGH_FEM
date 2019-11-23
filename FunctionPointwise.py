from fenics import *
def FunctionPointwise(V, bc_obs, pcoorx_obs, pcoory_obs, hp_obs):
     tol = DOLFIN_EPS

     def condicao_p0 (x): 
         return near(x[0], pcoorx_obs[0],tol) and near(x[1], pcoory_obs[0],tol)
     bcp_0 = DirichletBC(V, hp_obs[0], condicao_p0, method='pointwise')
     bc_obs.append(bcp_0)

     def condicao_p1 (x): 
         return near(x[0], pcoorx_obs[1],tol) and near(x[1], pcoory_obs[1],tol)
     bcp_1 = DirichletBC(V, hp_obs[1], condicao_p1, method='pointwise')
     bc_obs.append(bcp_1)

     def condicao_p2 (x): 
         return near(x[0], pcoorx_obs[2],tol) and near(x[1], pcoory_obs[2],tol)
     bcp_2 = DirichletBC(V, hp_obs[2], condicao_p2, method='pointwise')
     bc_obs.append(bcp_2)

     def condicao_p3 (x): 
         return near(x[0], pcoorx_obs[3],tol) and near(x[1], pcoory_obs[3],tol)
     bcp_3 = DirichletBC(V, hp_obs[3], condicao_p3, method='pointwise')
     bc_obs.append(bcp_3)

     def condicao_p4 (x): 
         return near(x[0], pcoorx_obs[4],tol) and near(x[1], pcoory_obs[4],tol)
     bcp_4 = DirichletBC(V, hp_obs[4], condicao_p4, method='pointwise')
     bc_obs.append(bcp_4)

     def condicao_p5 (x): 
         return near(x[0], pcoorx_obs[5],tol) and near(x[1], pcoory_obs[5],tol)
     bcp_5 = DirichletBC(V, hp_obs[5], condicao_p5, method='pointwise')
     bc_obs.append(bcp_5)

     return bc_obs