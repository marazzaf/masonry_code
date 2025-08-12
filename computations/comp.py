import numpy as np
import sys
sys.path.append('../utils/')
from graph import *
from energy import *
import matplotlib.pyplot as plt

#Fixing seed
np.random.seed(seed=136985)

#Material parameter for friction
s = 2

#Getting the points
d = 2 #Space dimension
N = 60 #20
pts = np.random.uniform(size=d*N)
points = pts.reshape((N,d))

#Creating the graph
GM = GranularMaterial(points, d, s)

##Plotting points
#GM.plot_graph()
#GM.plot_voronoi()
#sys.exit()

#Neumann condition on boundary edges
compression = 1 #1e2 #compressive force
eps = .1
stress_bnd = np.zeros((d, GM.Nbe))
for c1,c2 in GM.graph.edges:
    if GM.graph[c1][c2]['bnd']:
        id_e = GM.graph[c1][c2]['id_edge'] - GM.Ne
        normal = GM.graph[c1][c2]['normal']
        bary = GM.graph[c1][c2]['bary']
        if bary[1] > .9 and (bary[0] - .5) < .2:
            stress_bnd[:,id_e] = -normal
        else:
            stress_bnd[:,id_e] = -eps * normal

#Assembling the system to minimize the energy
E = Energy(GM, stress_bnd)

#Computing the normal stresses
f = E.solve(GM)
#sys.exit()

#Stress reconstruction
stress = stress_reconstruction(GM, stress_bnd, f)
file = VTKFile('sol.pvd')
for (i,s) in enumerate(stress):
    file.write(s,idx=i)
