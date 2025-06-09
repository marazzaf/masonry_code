import numpy as np
import sys
sys.path.append('../utils')
from graph import *
from energy import *
import matplotlib.pyplot as plt
#Material parameter for friction
s = 1

#Setting points in specific positions
d = 2 #Space dimension

#Getting the points
points = np.array([[1/4,1/4], [1/4,3/4], [3/4,1/4], [3/4,3/4]])

#Creating the graph
GM = GranularMaterial(points, d, s)

##Plotting points
#GM.plot_graph()
GM.plot_voronoi()

#Creating a force on the boundary cells
compression = 1e1 #compressive force
force_bnd = np.zeros((d,len(GM.bnd)))
i = 0
for c in GM.bnd:
    GM.graph.nodes[c]['id_cell'] = i
    force_bnd[:,i] = GM.pos_bary - GM.voronoi.points[c] #vector pointing towards the barycenter
    #plt.quiver(GM.voronoi.points[c][0], GM.voronoi.points[c][1], force_bnd[0,i], force_bnd[1,i]) #For plot
    i += 1
#plt.show()

#Assembling the system to minimize the energy
E = Energy(GM, force_bnd)
print(E.E)
sys.exit()

#Computing the forces
f = E.solve(d, GM.Nc)
sys.exit()

#Plotting the forces
voronoi_plot_2d(GM.voronoi)
for c1,c2 in GM.graph.edges:
    id_f = GM.graph[c1][c2]['id_edge']
    bary = GM.graph[c1][c2]['bary']
    plt.quiver(bary[0], bary[1], f[id_f][0], f[id_f][1])
plt.show()
