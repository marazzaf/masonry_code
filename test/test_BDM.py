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
V = VectorFunctionSpace(mesh, "BDM", 1)

#Testing interpolation
stress = Function(V, name='stress')
x = SpatialCoordinate(mesh)
stress.interpolate(outer(x,x))

#Test curl
W = FunctionSpace(mesh, 'DG', 0)
s1 = Function(W)
s1.interpolate(curl(stress[0,:]))
s2 = Function(W)
s2.interpolate(curl(stress[1,:]))

#Outputs
file = VTKFile('test.pvd')
file.write(s1,s2)
