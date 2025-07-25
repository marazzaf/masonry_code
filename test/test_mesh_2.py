import numpy as np
from firedrake.petsc import PETSc
from firedrake import *
from firedrake.output import VTKFile
import sys

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

#Space
V = FunctionSpace(mesh, 'RT', 1)

#Weak form
sigma = TrialFunction(V)
tau = TestFunction(V)
a = inner(div(sigma), div(tau)) * dx
L = Constant(0) * tau[0] * dx

#Impose weakly Dirichlet BC
n = FacetNormal(mesh)
h = CellDiameter(mesh)
eta = 1e2
#a += eta / h * inner(dot(sigma, n), dot(tau, n)) * ds
#a += eta * inner(dot(sigma, n), dot(tau, n)) * ds
g = Constant(1) #Dirichlet BC
#L = eta / h * inner(dot(tau, n), g) * ds
#L = eta * inner(dot(tau, n), g) * ds

#Dirichlet BC strong
x = SpatialCoordinate(mesh)
r = sqrt(x[0]*x[0] + x[1]*x[1])
normal = as_vector((x[0]/r, x[1]/r))
bcs = DirichletBC(V, normal, 'on_boundary')

# Solve
res = Function(V, name='res')
solve(a == L, res, bcs=bcs)

#test
test = assemble(dot(res, n) * ds) / assemble(1 * ds(mesh))
print(test)
#sys.exit()

#output
file = VTKFile('test.pvd')
file.write(res)
