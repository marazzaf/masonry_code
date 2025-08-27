import numpy as np
import sys
sys.path.append('../utils/')
from graph import *
from energy import *
import matplotlib.pyplot as plt

#Fixing seed
np.random.seed(seed=136985)

#Material parameter for friction
s = 10

#Getting the points
d = 2 #Space dimension
N = 60 #20
pts = np.random.uniform(size=d*N)
points = pts.reshape((N,d))

#Test
#points = np.array([[1/4,1/4], [1/4,3/4], [3/4,1/4], [3/4,3/4], [1/2,1/2]]) #Third verification test
points = np.array([[1/6,1/6], [1/6,5/6], [5/6,1/6], [5/6,5/6], [1/2,1/2]])
#points = np.array([[1/4,1/4], [1/4,3/4], [3/4,1/4], [3/4,3/4]]) #First verification test

#Creating the graph
GM = GranularMaterial(points, d, s)

##Plotting points
#GM.plot_graph()
#GM.plot_voronoi()
#sys.exit()

#Neumann condition on boundary edges
compression = 1 #1e2 #compressive force
eps = 1 #1 #.1
stress_bnd = np.zeros((d, GM.Nbe))
for c1,c2 in GM.graph.edges:
    if GM.graph[c1][c2]['bnd']:
        id_e = GM.graph[c1][c2]['id_edge'] - GM.Ne
        normal = GM.graph[c1][c2]['normal']
        #bary = GM.graph[c1][c2]['bary']
        stress_bnd[:,id_e] = -normal
        #if bary[1] > .9 and (bary[0] - .5) < .1:
        #    stress_bnd[:,id_e] = -compression * normal
        #else:
        #    stress_bnd[:,id_e] = -eps * normal

#Assembling the system to minimize the energy
E = Energy(GM, stress_bnd)

#Checking inequality constraints
sigma = -np.identity(2)
z = np.zeros(3*GM.Ne)
for c1,c2 in GM.graph.edges:
    if not GM.graph[c1][c2]['bnd']:
        normal = GM.graph[c1][c2]['normal']
        tangent = GM.graph[c1][c2]['tangent']
        length = GM.graph[c1][c2]['length']
        id_e = GM.graph[c1][c2]['id_edge']
        f = length * np.dot(sigma, normal)
        f_n = -np.dot(f,normal)
        f_t = np.dot(f,tangent)
        #f_tp = max(f_t, 0)
        #f_tn = max(-f_t,0)
        f_tp = s * length/2 #test
        f_tn = s * length/2 #test
        z[id_e] = f_n
        z[GM.Ne+2*id_e] = f_tp
        z[GM.Ne+2*id_e+1] = f_tn
        assert f_n > 0
        assert abs(f_t) < 1e-15

#Checking energy
zz = matrix(z)
print(zz)

#Checking fenchel-duality
test = E.G.T * zz + E.E
aux1 = test[:GM.d * GM.Nc] #normal components ok
print(aux1)

#test1 = (E.G.T * zz)[GM.d * GM.Nc:]
#test2 = E.E[GM.d * GM.Nc:]
#for t1,t2 in zip(test1,test2):
#    print(t1,t2)
#sys.exit()

aux2 = test[GM.d * GM.Nc:] #tangent components
print(aux2)

