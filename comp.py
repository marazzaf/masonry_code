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

#G.plot_graph()
G.plot_voronoi()
