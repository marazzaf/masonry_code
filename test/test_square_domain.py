import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../utils/')
from graph import *

#Fixing seed
np.random.seed(seed=136985)

#Numpy tolerance
eps = np.finfo(float).eps

#Space parameters
d = 2 #Space dimension
N = 20 #20

#Getting points inside a unit square
pts = np.random.uniform(size=d*N)
points = pts.reshape((N,d))

#Creating the granular material
GM = GranularMaterial(points, d)
#GM.plot_voronoi()
#sys.exit()
#GM.plot_graph()
sys.exit()

#Computing the Voronoi mesh
from scipy.spatial import Voronoi, voronoi_plot_2d
voronoi = Voronoi(points)

##Check which cells are on the boundary (they will have vertices outside the unit square)
#bnd = set()
#for c in range(len(voronoi.points)):
#    region = voronoi.point_region[c]
#    #Check if one vertex of the voronoi cell is outside the cube
#    for id_vert in voronoi.regions[region]:
#        pos_vert = voronoi.vertices[id_vert]
#        if pos_vert[0] < 0 or pos_vert[0] > 1 or pos_vert[1] < 0 or pos_vert[1] > 1:
#            bnd.add(c)
#            print(pos_vert)

##Compute edges of boundary cells
#G = GM.graph
#voronoi = GM.voronoi
#for c in GM.bnd:
#    if 
    

#plotting the Voronoi mesh
fig = voronoi_plot_2d(voronoi)
plt.xlim(0,1)
plt.ylim(0,1)
#plt.xlim(-5,5)
#plt.ylim(-5,5)
plt.show()
