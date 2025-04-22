import numpy as np
import sys
sys.path.append('../utils')
from graph import *
from energy import *

#Fixing seed
#np.random.seed(seed=136985)
np.random.seed(seed=1568798)

#Material parameter for friction
s = 1

#Space parameters
d = 2 #Space dimension
N = 5 #20

#Getting the points
L = 1 #N // 100
pts = np.random.uniform(size=d*N) * L
points = pts.reshape((N,d))

#Creating the GranularMaterial
GM = GranularMaterial(points, d)

#GM.plot_graph()
GM.plot_voronoi()
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

#Computing the foces
f = E.solve(d, GM.Ne)

#Tolerance for force-balance
eps = np.finfo(float).eps

##Checking global force balance
#G = GM.graph
#total_force = np.zeros(d)
##Checking first on boundary cells
#for c1 in GM.bnd:
#    id_cell = G.nodes[c1]['id_cell']
#    force_cell = force_bnd[:,id_cell] #Adding boundary force to the balance
#    for c2 in G.neighbors(c1): 
#        id_edge = G[c1][c2]['id_edge']
#        normal = G[c1][c2]['normal']
#        sign = np.dot(normal, GM.voronoi.points[c2] - GM.voronoi.points[c1])
#        sign /= abs(sign)
#        force_cell += sign * f[id_edge]
#    total_force += force_cell 
##Checking inner cells
#inner = set(range(GM.Nc)) - GM.bnd
#for c1 in inner:
#    force_cell = np.zeros(d)
#    for c2 in G.neighbors(c1): 
#        id_edge = G[c1][c2]['id_edge']
#        normal = G[c1][c2]['normal']
#        sign = np.dot(normal, GM.voronoi.points[c2] - GM.voronoi.points[c1])
#        sign /= abs(sign)
#        force_cell += sign * f[id_edge]
#    total_force += force_cell 
#    #print(G.nodes[c1]['bnd'])
#    #print(tot)
#
#try:
#    assert np.linalg.norm(total_force - total_ext_force) < eps
#except AssertionError:
#    print(np.linalg.norm(total_force - total_ext_force))

#Checking that force balance is true on each cell
G = GM.graph
#Checking first on boundary cells
for c1 in GM.bnd:
    id_cell = G.nodes[c1]['id_cell']
    force_cell = force_bnd[:,id_cell] #Adding boundary force to the balance
    for c2 in G.neighbors(c1): 
        id_edge = G[c1][c2]['id_edge']
        normal = G[c1][c2]['normal']
        sign = np.dot(normal, GM.voronoi.points[c2] - GM.voronoi.points[c1])
        sign /= abs(sign)
        force_cell += sign * f[id_edge]
#        print(sign,f[id_edge])
    #try:
    #    assert np.linalg.norm(force_cell) < eps
    #except AssertionError:
    #    print('bnd')
    print(np.linalg.norm(force_cell))
        
#Checking inner cells
inner = set(range(GM.Nc)) - GM.bnd
for c1 in inner:
    force_cell = np.zeros(d)
    for c2 in G.neighbors(c1): 
        id_edge = G[c1][c2]['id_edge']
        normal = G[c1][c2]['normal']
        sign = np.dot(normal, GM.voronoi.points[c2] - GM.voronoi.points[c1])
        sign /= abs(sign)
        force_cell += sign * f[id_edge]
        print(sign,f[id_edge])
    #try:
    #    assert np.linalg.norm(force_cell) < eps
    #except AssertionError:
    #print(np.linalg.norm(force_cell))
    print(force_cell)
    #sys.exit()
