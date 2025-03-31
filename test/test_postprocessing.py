import numpy as np
import sys
sys.path.append('../utils')
from graph import *
from energy import *

#Fixing seed
#np.random.seed(seed=136985)
#np.random.seed(seed=1568798) #Pb between 502 and 503

#Material parameter for friction
s = 1

#Space parameters
d = 2 #Space dimension
N = 500 #20

#Getting the points
L = 1 #N // 100
pts = np.random.uniform(size=d*N) * L
points = pts.reshape((N,d))

##Test
#aux = np.linspace(0, 1, N, endpoint=True)
#from itertools import product
#points = list(product(aux, aux))

#Creating the GranularMaterial
GM = GranularMaterial(points, d, s, L)

#GM.plot_graph()
#GM.plot_voronoi()
#sys.exit()

#Creating a force on the boundary cells
force_bnd = np.zeros((d,len(GM.bnd)))
i = 0
for c in GM.bnd:
    GM.graph.nodes[c]['id_cell'] = i
    force_bnd[:,i] = GM.pos_bary - GM.voronoi.points[c] #vector pointing towards the barycenter
    i += 1
total_ext_force = force_bnd.sum(axis=1)

#Assembling the system to minimize the energy
E = Energy(GM, force_bnd)

#Computing the forces
f = E.solve(d, GM.Ne)

#Postprocessing

#Plot
import matplotlib.pyplot as plt
x = np.zeros(GM.Nc)
y = np.copy(x)
z = np.copy(x)
i = 0
#Computing the ratios
for c1 in GM.graph.nodes:
    l1, l2, linf = 0,0,0
    ratio = 0
    for c2 in GM.graph.neighbors(c1):
        id_edge = GM.graph[c1][c2]['id_edge']
        l1 += np.linalg.norm(f[id_edge])
        l2 += np.linalg.norm(f[id_edge])**2
        linf = max(linf, np.linalg.norm(f[id_edge]))
        ratio = l2 / l1 / linf
    #print(GM.graph.nodes[c1]['bnd'])
    #print(ratio)
    pos = GM.voronoi.points[c1]
    x[i] = pos[0]
    y[i] = pos[1]
    z[i] = ratio
    i += 1

#plot
plt.scatter(x, y, c=z, cmap='jet', edgecolor='k', s=50)
plt.colorbar()
plt.show()
