import numpy as np
import sys
sys.path.append('../../utils')
from graph_aux2 import *
from shapely.geometry import box
#from energy import *
#from reconstructions import stress_reconstruction

#Setting points in specific positions
d = 2 #Space dimension

#Create shapely geometry of a unit square
unit_square = box(0.0, 0.0, 1.0, 1.0)
geometry = GeometryCollection([unit_square])

#Getting the points in the geometry
points = np.array([[.25, .25], [.25,.75], [.75,.25], [.75,.75], [.5, .5]])

#Creating the graph
GM = GranularMaterial(points, geometry)

#Plotting points
GM.plot_voronoi()
GM.plot_mesh()
