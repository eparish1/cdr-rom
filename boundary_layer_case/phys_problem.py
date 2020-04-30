from dolfin import *
import numpy as np
import os
import sys
sys.path.append('../cdr_common/')
from postProcessor import *
from romProblemClass import romProblemClass
from physProblemClass import physProblemClass
from cdr_rom import *
if has_linear_algebra_backend("Epetra"):
    parameters["linear_algebra_backend"] = "Epetra"

##============================================
# Physical problem for traveling L wave
class x_velocity(UserExpression):
  def eval(self,value,x):
     value[0] = 0.5*cos(pi/3.)

class y_velocity(UserExpression):
  def eval(self,value,x):
     value[0] = 0.5*sin(pi/3.)

class InitialCondition(UserExpression):
    def eval(self, value, x):
      value[0] = 0.

def buildPhysProblem(femProblem):
  ## Define general terms
  f_mag = 1.
  nu_mag = 1e-3
  sigma_mag = 1.
  u_field = Function(femProblem.functionSpace)
  u_field.interpolate(x_velocity())
  v_field = Function(femProblem.functionSpace)
  v_field.interpolate(y_velocity())
  physProblem = physProblemClass(nu_mag,sigma_mag,f_mag,u_field,v_field,InitialCondition)
  return physProblem
#============================================

