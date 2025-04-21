import numpy as np
import sys
sys.path.append('../utils')
from graph import *
from energy import *

#Fixing seed
np.random.seed(seed=136985)

#Material parameter for friction
s = 1

#Setting points in specific positions
d = 2 #Space dimension
N = 9

#Getting the points
points = np.array([[1/6,1/6], [1/2,1/6], [5/6,1/6], [1/6,1/2], [1/2,1/2], [5/6,1/2], [1/6,5/6], [1/2,5/6], [5/6,5/6]])

GM = GranularMaterial(points, d, s)

#GM.plot_graph()
#GM.plot_voronoi()

#Creating a force on the boundary cells
compression = 1e2 #compressive force
force_bnd = np.zeros((d,len(GM.bnd)))
i = 0
for c in GM.bnd:
    GM.graph.nodes[c]['id_cell'] = i
    force_bnd[:,i] = GM.pos_bary - GM.voronoi.points[c] #vector pointing towards the barycenter
    #pos = GM.voronoi.points[c]
    #if pos[0] < .2:
    #    force_bnd[:,i] = np.array([1, 0])
    #elif pos[0] > .8:
    #    force_bnd[:,i] = -np.array([1, 0])
    #if pos[1] < .2:
    #    force_bnd[:,i] += np.array([0, 1])
    #elif pos[1] > .8:
    #    force_bnd[:,i] += compression * np.array([0, -1])
    i += 1

#Assembling the system to minimize the energy
E = Energy(GM, force_bnd)

#Computing the forces
f = E.solve(d, GM.Ne)

#Plotting the forces
import matplotlib.pyplot as plt
voronoi_plot_2d(GM.voronoi)
for c1,c2 in GM.graph.edges:
    id_f = GM.graph[c1][c2]['id_edge']
    bary = GM.graph[c1][c2]['bary']
    plt.quiver(bary[0], bary[1], f[id_f][0], f[id_f][1])
plt.show()
