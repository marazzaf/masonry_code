import numpy as np
import sys
sys.path.append('../utils')
from graph import *
from energy import *
import matplotlib.pyplot as plt
#import pyvoro

#Material parameter for friction
s = 1

#Setting points in specific positions
d = 2 #Space dimension

#Getting the points
points = np.array([[1/4,1/4], [1/4,3/4], [3/4,1/4], [3/4,3/4]])

##Voronoi mesh creation
#voronoi = pyvoro.compute_2d_voronoi(
#    points.tolist(),                        # seed points
#    [[0, 1], [0, 1]],                      # bounding box
#    0.1                                    # block size ≈ sqrt(cell area)
#)

#Creating the graph
GM = GranularMaterial(points, d, s)

##Plotting points
#GM.plot_graph()
GM.plot_voronoi()
sys.exit()

#Creating a force on the boundary cells
compression = 1e2 #compressive force
force_bnd = np.zeros((d,len(GM.bnd)))
i = 0
for c in GM.bnd:
    GM.graph.nodes[c]['id_cell'] = i
    force_bnd[:,i] = compression * (GM.pos_bary - GM.graph.nodes[c]['pos']) #vector pointing towards the barycenter
    #plt.quiver(GM.voronoi.points[c][0], GM.voronoi.points[c][1], force_bnd[0,i], force_bnd[1,i]) #For plot
    i += 1
#plt.show()

#Assembling the system to minimize the energy
E = Energy(GM, force_bnd)
#print(E.E)

#Computing the forces
f = E.solve(d, GM.Nc)
print(f)
sys.exit()

#Plotting the forces
voronoi_plot_2d(GM.voronoi)
for c in GM.graph.nodes:
    bary = GM.graph.nodes[c]['bary']
    plt.quiver(bary[0], bary[1], f[c,0], f[c,1])
plt.show()
