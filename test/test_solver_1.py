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

#Creating the graph
GM = GranularMaterial(points, d, s)
#print(GM.graph.edges(data=True))
#sys.exit()

##Plotting points
#GM.plot_graph()
#GM.plot_voronoi()

#Neumann condition on boundary edges
compression = 1e-7 #1e2 #compressive force
stress_bnd = np.zeros((d, GM.Nbe))
for c1,c2 in GM.graph.edges:
    if GM.graph[c1][c2]['bnd']:
        id_e = GM.graph[c1][c2]['id_edge'] - GM.Ne
        normal = GM.graph[c1][c2]['normal']
        stress_bnd[:,id_e] = -compression * normal
        #plt.quiver(GM.graph[c1][c2]['bary'][0], GM.graph[c1][c2]['bary'][1], stress_bnd[0,id_e], stress_bnd[1,id_e], color='red') #For plot
#plt.show()

#print(stress_bnd)
#sys.exit()

#Assembling the system to minimize the energy
E = Energy(GM, stress_bnd)
#print(E.E)
#sys.exit()

#Computing the forces
f = E.solve(GM)
print(f)
#sys.exit()

#Plotting the forces
for c1,c2 in GM.graph.edges:
    if not GM.graph[c1][c2]['bnd']:
        bary = GM.graph[c1][c2]['bary']
        id_e = GM.graph[c1][c2]['id_edge']
        n = GM.graph[c1][c2]['normal']
        plt.quiver(bary[0], bary[1], n[0], n[1], color='blue')
        t = GM.graph[c1][c2]['tangent']
        #plt.quiver(bary[0], bary[1], t[0], t[1])
        plt.quiver(bary[0], bary[1], f[id_e,0], f[id_e,1])
plt.show()
