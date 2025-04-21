import numpy as np
import sys
sys.path.append('./utils')
from graph import *
from energy import *

#Fixing seed
np.random.seed(seed=136985)

#Material parameter for friction
s = 1

#Getting the points
d = 2 #Space dimension
N = 20 #20
pts = np.random.uniform(size=d*N)
points = pts.reshape((N,d))

GM = GranularMaterial(points, d, s)

#Test
import matplotlib.pyplot as plt
voronoi_plot_2d(GM.voronoi)
for c in GM.bnd:
    pos = GM.voronoi.points[c]
    plt.plot(pos[0], pos[1], 'ro')
plt.show() 

#GM.plot_graph()
#GM.plot_voronoi()

#Creating a force on the boundary cells
force_bnd = np.zeros((d,len(GM.bnd)))
i = 0
for c in GM.bnd:
    GM.graph.nodes[c]['id_cell'] = i
    force_bnd[:,i] = GM.pos_bary - GM.voronoi.points[c] #vector pointing towards the barycenter
    i += 1

#Assembling the system to minimize the energy
E = Energy(GM, force_bnd)

#Computing the forces
f = E.solve(d, GM.Ne)
