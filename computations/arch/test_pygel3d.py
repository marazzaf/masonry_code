import numpy as np
import sys
sys.path.append('../../utils')
from graph_aux2 import *
#from energy import *
#from reconstructions import stress_reconstruction

#Material parameter for friction
s = 1

#Setting points in specific positions
d = 2 #Space dimension

#Getting the points
points = np.array([[.25, .25], [.25,.75], [.75,.25], [.75,.75], [.5, .5]])

#Creating the graph
GM = GranularMaterial(points, d, s)

#Plotting points
#GM.plot_voronoi()
GM.plot_mesh()
