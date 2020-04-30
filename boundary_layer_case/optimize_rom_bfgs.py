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





def optimizeParams(tauOptimize):
 romProblem.tau = tauOptimize
 error_save_l2,error_save_h1 =  executeRom(romProblem,physProblem,pod_basis['UDOFSave'],dt_fom)
 if objective == 'L2':
   return error_save_l2 
 if objective == 'H1':
   return error_save_h1 


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













