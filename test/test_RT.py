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
W = FunctionSpace(mesh, 'DG', 0)
Z = V * W

#Testing interpolation
stress = Function(V, name='stress')
x = SpatialCoordinate(mesh)
#aux = as_vector((-x[1], x[0]))
BC = Constant(((-1, -1), (-1, -1)))
stress.interpolate(BC)
#VV = VectorFunctionSpace(mesh, 'DG', 0)
#aux = Function(VV)
#aux.interpolate(div(stress))

##Output
#file = VTKFile('test_3.pvd')
#file.write(aux)

#Output
file = VTKFile('test.pvd')
file.write(stress)
#print(curl(stress[:,0]).at((.25,.25)))
#sys.exit()

#Testing BC
#bc = DirichletBC(V, -Identity(2), 'on_boundary')
#bc = DirichletBC(Z.sub(0), -Identity(2), 'on_boundary')
#x = SpatialCoordinate(mesh)
#bc = DirichletBC(Z.sub(0), Constant(((-1, -1), (-1, -1))), 'on_boundary')
bc = DirichletBC(Z.sub(0), BC, 'on_boundary')
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
#L = x[0] * tau[0,0] * dx

#Solve
res = Function(Z, name='res')
params = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type' : 'mumps'}
solve(a == L, res, bcs=bc, solver_parameters=params)
stress = res.sub(0)

#val = stress.at((.75,.25))
#print(val[0,1] - val[1,0])

file = VTKFile('test_2.pvd')
file.write(stress,res.sub(1))
