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
#print(GM.voronoi.regions)
#print(GM.Nc)
#GM.plot_voronoi()
#sys.exit()
#GM.plot_graph()
#sys.exit()

for i, list_vert in enumerate(GM.voronoi.regions):
    if -1 in list_vert:
        print(i)

sys.exit()

#plotting the Voronoi mesh
fig = voronoi_plot_2d(GM.voronoi)
#plt.xlim(0,1)
#plt.ylim(0,1)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.show()
