import numpy as np
from firedrake import *
from firedrake.petsc import PETSc

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

# === Wrap into Firedrake mesh ===
mesh = Mesh(plex)

# === Now you can create FunctionSpaces ===
V = FunctionSpace(mesh, "CG", 1)

# Define a function and interpolate an expression
u = Function(V, name='u')
x, y = SpatialCoordinate(mesh)
u.interpolate(x*y)

# Plot to verify
from firedrake.output import VTKFile
file = VTKFile('test.pvd')
file.write(u)
