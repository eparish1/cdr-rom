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



N = 32
p = 2
dt = 0.001
et = 5.
methodContinuous = 'SUPG'
methodDiscrete = 'APG'

femProblem  = femProblemClass(N,p,methodContinuous,dt,et,0.01)
physProblem = buildPhysProblem(femProblem)

##== Make ROM Problem
basis_type = 'L2'
pod_basis = np.load('solfomNONE_tau_analytic_N_66049_p_2/pod_basis.npz')
if (basis_type == 'L2'):
  Phi = pod_basis['Phi']
  sig = pod_basis['sigma']
if (basis_type == 'H1'):
  Phi = pod_basis['PhiH']
  sig = pod_basis['sigmaH']
sig = np.cumsum(sig) / np.sum(sig)
K = np.size(sig[sig<=0.99999])
#K = 5
Phi = Phi[:,0:K]
print(np.shape(Phi))
romProblem = romProblemClass(femProblem,Phi,basis_type,methodContinuous,methodDiscrete,dt,et,0.01)
#===========


dt_fom = 0.001
#tau_array =  np.logspace(-4,-1.,20)
tau_array_rom = np.array([0.,10**-5,10**-4,2.5*10**-4,5*10**-4,10**-3,2*10**-3,3*10**-3,4*10**-3,5*10**-3,6*10**-3,7*10**-3,8*10**-3,9*10**-3,10**-2,1.5*10**-2,2*10**-2,2.5*10**-2,3*10**-2,4*10**-2,5*10**-2])
tau_array_fom = np.array([0.,10**-5,10**-4,2.5*10**-4,5*10**-4,10**-3,2*10**-3,3*10**-3,4*10**-3,5*10**-3,6*10**-3,7*10**-3,8*10**-3,9*10**-3,10**-2,1.5*10**-2,2*10**-2,2.5*10**-2,3*10**-2,4*10**-2,5*10**-2])

for tauFEM in tau_array_fom: 
  for tauROM in tau_array_rom:
    romProblem.tau = tauROM*1.
    romProblem.femProblem.tau = tauFEM*1.
    executeRom(romProblem,physProblem,pod_basis['UDOFSave'],dt_fom)
