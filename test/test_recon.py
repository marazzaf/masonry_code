import numpy as np
import sys
sys.path.append('../utils')
from graph import *
from energy import *
import matplotlib.pyplot as plt
from reconstructions import stress_reconstruction

#Material parameter for friction
s = 10

#Space dimension
d = 2 

#Getting the points
points = np.array([[1/4,1/4], [1/4,3/4], [3/4,1/4], [3/4,3/4]])

#Creating the graph
GM = GranularMaterial(points, d, s)

#plot
#GM.plot_voronoi()
#GM.plot_graph()
#sys.exit()

#Neumann condition on boundary edges
compression = 1 #1e2 #compressive force
eps = 1
S = -compression * np.array([[1, eps], [eps,1]])
stress_bnd = np.zeros((d, GM.Nbe))
for c1,c2 in GM.graph.edges:
    if GM.graph[c1][c2]['bnd']:
        id_e = GM.graph[c1][c2]['id_edge'] - GM.Ne
        normal = GM.graph[c1][c2]['normal']
        stress_bnd[:,id_e] = np.dot(S, normal)

#Assembling the system to minimize the energy
E = Energy(GM, stress_bnd)

#Computing the normal stresses
f = E.solve(GM)

#Stress reconstruction
stress = stress_reconstruction(GM, stress_bnd, f)
sys.exit()
