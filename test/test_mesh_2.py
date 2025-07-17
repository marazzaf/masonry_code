import numpy as np
from firedrake.petsc import PETSc
from firedrake import *
from firedrake.output import VTKFile

# === Step 1: Create hexagon vertices (radius = 1) ===
angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 angles from 0 to 5π/3
hex_vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1)

# === Step 2: Add barycentre at (0,0) ===
coordinates = np.vstack([[0.0, 0.0], hex_vertices])  # vertex 0 is center

# === Step 3: Define 6 triangles using barycentre and adjacent vertices ===
cells = []
for i in range(6):
    v1 = i + 1
    v2 = ((i + 1) % 6) + 1
    cells.append([0, v1, v2])  # triangle: [center, vertex i, vertex i+1]
cells = np.array(cells, dtype='int32')

# === Create DMPlex mesh with topology ===
plex = PETSc.DMPlex().createFromCellList(2, cells, coordinates, interpolate=True)

# === Wrap into Firedrake mesh ===
mesh = Mesh(plex)

##Mixed space
#V = FunctionSpace(mesh, 'RT', 1)
#W = FunctionSpace(mesh, 'DG', 1)
#Z = V * W
#
##Weak form
#sigma, xi = TrialFunctions(Z)
#tau, zeta = TestFunctions(Z)
#a = inner(div(sigma), zeta) * dx
#L = Constant(0) * tau[0] * dx

#Space
V = FunctionSpace(mesh, 'RT', 1)

#Weak form
sigma = TrialFunction(V)
tau = TestFunction(V)
a = inner(div(sigma), div(tau)) * dx
L = Constant(0) * tau[0] * dx

#Dirichlet BC
zero = Constant((0, 0))
n = FacetNormal(mesh)
x = SpatialCoordinate(mesh)
N = Function(V).interpolate(n)
bc = DirichletBC(V, N, "on_boundary") #x
#bc = DirichletBC(Z.sub(0), x, 'on_boundary')

# Solve
res = Function(V)
solve(a == L, res, bcs=bc)
#res = Function(Z)
#v_basis = VectorSpaceBasis(constant=True)
#nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), v_basis])
#solve(a == L, res, bcs=bc, nullspace=nullspace)

#output
file = VTKFile('test.pvd')
file.write(res)

## Plot to check
#import matplotlib.pyplot as plt
#firedrake.triplot(mesh)
#plt.gca().set_aspect('equal')
#plt.title("Manually Created Triangle Mesh")
#plt.show()
