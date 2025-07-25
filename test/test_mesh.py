import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
import sys

# === Define geometry ===
coordinates = np.array([
    [0.0, 0.0],  # vertex 0
    [1.0, 0.0],  # vertex 1
    [0.0, 1.0],  # vertex 2
    [1.0, 1.0],  # vertex 3
])

# Two triangles: (0,1,2) and (1,3,2)
cells = np.array([
    [0, 1, 2],
    [1, 3, 2]
], dtype='int32')

# === Create DMPlex mesh with topology ===
plex = PETSc.DMPlex().createFromCellList(2, cells, coordinates, interpolate=True)

#Mark boundary facets
boundary_facets = []
f_start, f_end = plex.getHeightStratum(1)
for f in range(f_start, f_end):
    support = plex.getSupport(f)
    if len(support) == 1:
        boundary_facets.append(f)

# Sort by order they appear
for idx, facet in enumerate(boundary_facets):
    plex.setLabelValue('bnd', facet, idx)

# === Wrap into Firedrake mesh ===
mesh = Mesh(plex)

# === Now you can create FunctionSpaces ===
V = FunctionSpace(mesh, "CG", 1)

#Weak form
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx
L = Constant(0) * v * dx

#Creating the Dirichet BC
bcs = []
g = Constant(1)
#Loop on bnd facets
for idx, f in enumerate(boundary_facets):
    vertices = plex.getCone(f)
    print(vertices)
    #Need to compute the outwards normal from the vertices?
    bc = DirichletBC(V, g, idx)
    bcs.append(bc)

sys.exit()
# Solve
res = Function(V, name='res')
solve(a == L, res, bcs=bcs)

#output
file = VTKFile('test.pvd')
file.write(res)
