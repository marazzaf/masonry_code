from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from firedrake.output import VTKFile

# Create a unit triangle mesh
#mesh = UnitTriangleMesh()
N = 1
mesh = UnitSquareMesh(N, N, diagonal='crossed')

#FunctionSpace
V = VectorFunctionSpace(mesh, "RT", 1)
W = FunctionSpace(mesh, 'DG', 0) #CG 1?
Z = V * W

##Testing interpolation
#stress = Function(V, name='stress')
#stress.interpolate(Identity(2))
stress = Function(Z, name='stress')

##Output
#file = VTKFile('test.pvd')
#file.write(stress)

#Testing BC
#bc = DirichletBC(V, -Identity(2), 'on_boundary')
bc = DirichletBC(Z.sub(0), -Identity(2), 'on_boundary')
#bc.apply(stress.vector())

##Output
#file = VTKFile('test.pvd')
#file.write(stress)

#Weak form
#sigma = TrialFunction(V)
#tau = TestFunction(V)
sigma,p = TrialFunctions(Z)
tau,q = TestFunctions(Z)
a = inner(div(sigma), div(tau)) * dx #LS for div equation
mat_p = as_matrix(((0, p), (-p, 0)))
mat_q = as_matrix(((0, q), (-q, 0)))
a += inner(sigma, mat_q) * dx #Constraints
a += inner(tau, mat_p) * dx #Constraints

#Linear form
L = Constant(0) * tau[0,0] * dx

#Solve
res = Function(V, name='stress')
params = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type' : 'mumps'}
solve(a == L, stress, bcs=bc, solver_parameters=params)
stress,lag = stress.split()

print(stress.at((.75,.25)))

file = VTKFile('test.pvd')
file.write(stress)
