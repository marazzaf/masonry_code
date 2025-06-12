import numpy as np
import sys
sys.path.append('./utils')
from graph import *
from energy import *
import matplotlib.pyplot as plt

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

##Plotting points
#GM.plot_graph()
#GM.plot_voronoi()

#Creating a force on the boundary cells
compression = 1e1 #compressive force
force_bnd = np.zeros((d,len(GM.bnd)))
i = 0
##Anisitrpoic force
#for c in GM.bnd:
#    GM.graph.nodes[c]['id_cell'] = i
#    pos = GM.graph.nodes[c]['pos']
#    if pos[0] < .2:
#        force_bnd[:,i] = np.array([1, 0])
#    elif pos[0] > .8:
#        force_bnd[:,i] = -np.array([1, 0])
#    if pos[1] < .2:
#        force_bnd[:,i] += np.array([0, 1])
#    elif pos[1] > .8:
#        force_bnd[:,i] += compression * np.array([0, -1])
#    plt.quiver(GM.graph.nodes[c]['pos'][0], GM.graph.nodes[c]['pos'][1], force_bnd[0,i], force_bnd[1,i], color='red') #For plot
#    i += 1
#plt.show()

#Isotropic force
for c in GM.bnd:
    GM.graph.nodes[c]['id_cell'] = i
    force_bnd[:,i] = compression * (GM.pos_bary - GM.graph.nodes[c]['pos']) #vector pointing towards the barycenter
    plt.quiver(GM.graph.nodes[c]['pos'][0], GM.graph.nodes[c]['pos'][1], force_bnd[0,i], force_bnd[1,i], color='red') #For plot
    i += 1

#Assembling the system to minimize the energy
E = Energy(GM, force_bnd)

#Computing the forces
f = E.solve(d, GM.Nc)
print(f)

#Plotting the forces
for c in GM.graph.nodes:
    bary = GM.graph.nodes[c]['pos']
    plt.quiver(bary[0], bary[1], f[c,0], f[c,1])
plt.show()
