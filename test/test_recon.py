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
#print(GM.voronoi[0]['faces'])
#print(GM.voronoi[1]['faces'])
#print(GM.graph.nodes[1]['face_dict'])
#print(GM.voronoi[2]['faces'])
#print(GM.voronoi[3]['faces'])
#GM.plot_graph()

#Neumann condition on boundary edges
compression = 1 #compressive force
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

print(f)

#Stress reconstruction
stress = stress_reconstruction(GM, stress_bnd, f)
file = VTKFile('test.pvd')
for (i,s) in enumerate(stress):
    file.write(s,idx=i)
