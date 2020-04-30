from dolfin import *
import numpy as np
import os
import sys
sys.path.append('../cdr_common/')
from postProcessor import *
from romProblemClass import romProblemClass
from physProblemClass import physProblemClass
from cdr_fom import *
from phys_problem import *
from femProblemClass import femProblemClass



N = 32
p = 2
Nc= 32
methodContinuous = 'NONE'
dt = 1.0
et = 5.0
femProblem  = femProblemClass(N,p,methodContinuous,dt,et,0.)
femCoarseProblem  = femProblemClass(Nc,p,methodContinuous,dt,et,0.)
physProblem = buildPhysProblem(femProblem)
executeFom(dt,femProblem,physProblem,femCoarseProblem)
