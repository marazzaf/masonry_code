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
points = np.array([[1/6,1/6], [1/2,1/6], [5/6,1/6], [1/6,1/2], [1/2,1/2], [5/6,1/2], [1/6,5/6], [1/2,5/6], [5/6,5/6]])

GM = GranularMaterial(points, d, s)

##Plotting points
#GM.plot_graph()
#GM.plot_voronoi()

#Creating a force on the boundary cells
compression = 1e2 #compressive force
force_bnd = np.zeros((d,len(GM.bnd)))
i = 0
for c in GM.bnd:
    GM.graph.nodes[c]['id_cell'] = i
    force_bnd[:,i] = compression * (GM.pos_bary - GM.graph.nodes[c]['pos']) #vector pointing towards the barycenter
    plt.quiver(GM.graph.nodes[c]['pos'][0], GM.graph.nodes[c]['pos'][1], force_bnd[0,i], force_bnd[1,i], color='red') #For plot
    i += 1

#Assembling the system to minimize the energy
E = Energy(GM, force_bnd)

#Computing the forces
f = E.solve(d, GM.Nc, GM.Ne)
print(f)

#Plotting the forces
for c in GM.graph.nodes:
    bary = GM.graph.nodes[c]['pos']
    plt.quiver(bary[0], bary[1], f[c,0], f[c,1])
plt.show()
