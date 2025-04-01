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
#print(GM.voronoi.vertices)
GM.plot_graph()
#GM.plot_voronoi()


##plotting the Voronoi mesh
#fig = voronoi_plot_2d(GM.voronoi)
##plt.xlim(0,1)
##plt.ylim(0,1)
#plt.xlim(-0.2,1.2)
#plt.ylim(-0.2,1.2)
#plt.show()
