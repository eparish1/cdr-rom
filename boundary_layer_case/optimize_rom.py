from dolfin import *
import numpy as np
import os
import sys
sys.path.append('../cdr_common/')
from postProcessor import *
from romProblemClass import romProblemClass
from physProblemClass import physProblemClass
from femProblemClass import femProblemClass

from cdr_rom import *
from phys_problem import *
if has_linear_algebra_backend("Epetra"):
    parameters["linear_algebra_backend"] = "Epetra"


N = 32
p = 2
dt = 0.001
et = 5.
methodContinuous = 'NONE'
methodDiscrete = 'APG'
femProblem  = femProblemClass(N,p,methodContinuous,dt,et)
physProblem = buildPhysProblem(femProblem)





def optimizeParams(romProblem,physProblem,pod_basis,dt_fom):
 #dt_ratio = np.array([1.,2.,5.,10.,20.,50.,100.])
 #dtBigger = np.array([1./20.,1./10.,1./5.,1./2.])
 #dt_ratio = np.append(dtBigger,dt_ratio)
 #dt_array = 0.001/dt_ratio
 tau_array = np.array([10**-4,2.5*10**-4,5*10**-4,10**-3,2*10**-3,3*10**-3,4*10**-3,5*10**-3,6*10**-3,7*10**-3,8*10**-3,9*10**-3,10**-2,1.5*10**-2,2*10**-2,2.5*10**-2,3*10**-2,4*10**-2,5*10**-2])
 nsamps = int( np.size(tau_array) )
 error_save_l2 = np.zeros(nsamps)
 error_save_h1 = np.zeros(nsamps)
 for i in range(0,nsamps):
   error_save_l2[i],error_save_h1[i] =   executeRom(tau_array[i],romProblem,physProblem,pod_basis['UDOFSave'],dt_fom)
   print('tau = ' + str(tau_array[i]), ' l2 error = ' + str(error_save_l2[i]), ' h1 error = ' + str(error_save_h1[i]))

 best_index_l2 = np.argmin(error_save_l2)
 best_index_h1 = np.argmin(error_save_h1)
 if objective == 'L2':
   return tau_array[best_index_l2],error_save_l2[best_index_l2],error_save_h1[best_index_l2]
 if objective == 'H1':
   return tau_array[best_index_h1],error_save_l2[best_index_h1],error_save_h1[best_index_h1]


# Create mesh and define function space
pod_basis = np.load('solfomNONE_tau_analytic_N_66049_p_2/pod_basis.npz')
objective = 'L2'
basis_type = 'L2'


error_save_l2 = np.zeros(0)
error_save_h1 = np.zeros(0)
tau_opt = np.zeros(0)
dt_fom = 0.001
for K in range(5,6):
  print('==================')
  print('K = ' + str(K))
  tau = np.array(1)*0.0001
  if (basis_type == 'L2'):
    Phi = pod_basis['Phi'][:,0:K]
  if (basis_type == 'H1'):
    Phi = pod_basis['PhiH'][:,0:K]

  romProblem = romProblemClass(femProblem,Phi,basis_type,methodContinuous,methodDiscrete,dt,et)
  tau,error_l2,error_h1 = optimizeParams(romProblem,physProblem,pod_basis,dt_fom)
  error_save_l2 = np.append(error_save_l2,error_l2)
  error_save_h1 = np.append(error_save_h1,error_h1)
  tau_opt = np.append(tau_opt,tau)
  np.savez('tau_opt2_' + basis_type + '_' + objective + '_' + methodContinuous + '_' + methodDiscrete,tau_opt=tau_opt,error_l2=error_save_l2,error_h1=error_save_h1)













