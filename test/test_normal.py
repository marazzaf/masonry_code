from firedrake import *
from firedrake.output import VTKFile

N = 5
mesh = UnitSquareMesh(N,N)

#V = VectorFunctionSpace(mesh, 'CG', 1)
V = FunctionSpace(mesh, 'RT', 1)

u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx
L = Constant(0) * v[0] * dx

#Impose weakly BC
n = FacetNormal(mesh)
h = CellDiameter(mesh)
eta = 1
a += eta / h * inner(dot(u, n), dot(v, n)) * ds
g = Constant(1) #Dirichlet BC
L += eta / h * inner(dot(v, n), g) * ds


res = Function(V)
solve(a == L, res)

file = VTKFile('test.pvd')
file.write(res)
