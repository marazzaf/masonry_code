import numpy as np
import sys
sys.path.append('./utils')
from graph import *

#Fixing seed
np.random.seed(seed=136985)

#Material parameter for friction
s = 1

#Getting the points
d = 2 #Space dimension
N = 20 #20
pts = np.random.uniform(size=d*N)
points = pts.reshape((N,d))

G = GranularMaterial(points, d)

#Checking that force balance is true on each cell
for c1 in G.nodes:
    tot = np.zeros(d)
    if G.nodes[c1]['bnd']: #bnd particle
        id_cell = G.nodes[c1]['id_cell']
        tot += force_bnd[:,id_cell] #Adding boundary force to the balance
    for c2 in G.neighbors(c1): 
        id_edge = G[c1][c2]['id_edge']
        normal = n[id_edge]
        sign = np.dot(normal, vor.points[c2] - vor.points[c1])
        sign /= abs(sign)
        tot += sign * f[id_edge]
    #print(G.nodes[c1]['bnd'])
    #print(tot)

#print(force_bnd.sum(axis=1))
#sys.exit()
