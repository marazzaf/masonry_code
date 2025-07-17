import numpy as np
from firedrake.petsc import PETSc
from firedrake import *

# Vertices: triangle with points (0,0), (1,0), (0,1)
coordinates = np.array([
    [0.0, 0.0],  # vertex 0
    [1.0, 0.0],  # vertex 1
    [0.0, 1.0]   # vertex 2
])

# One triangle made of the 3 vertices
cells = np.array([
    [0, 1, 2]
], dtype='int32')

# === Create DMPlex mesh with topology ===
plex = PETSc.DMPlex().createFromCellList(2, cells, coordinates, interpolate=True)

# === Wrap into Firedrake mesh ===
mesh = Mesh(plex)

#test
V = FunctionSpace(mesh, 'CG', 1)

f = Function(V)
x = SpatialCoordinate(mesh)
f.interpolate(x[0])

from firedrake.output import VTKFile
file = VTKFile('test.pvd')
file.write(f)

# Plot to check
import matplotlib.pyplot as plt
firedrake.triplot(mesh)
plt.gca().set_aspect('equal')
plt.title("Manually Created Triangle Mesh")
plt.show()
