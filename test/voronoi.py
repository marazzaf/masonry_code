import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
import sys

#Fixing seed
np.random.seed(seed=136985)

#Getting the points
N = 200 #10
pts = uniform.rvs(size=2*N)
points = pts.reshape((N,2))

#Test
#points = np.array([[.25, .25], [.75, .25], [.25, .75], [.75, .75]])
#points = np.array([[.25, .25], [.75, .25], [.25, .75], [.75, .75], [.5, .5]])

#Computing the Voronoi mesh
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(points)

#Determining the cells that are not closed
list_open_regions = []
for id in range(len(vor.regions)):
    if -1 in vor.regions[id]:
        list_open_regions.append(id)
print(list_open_regions)
index_open_cells = [] #list of indices of input points that have open cells
for id_reg in list_open_regions:
    id_pt = np.where(vor.point_region == id_reg)[0][0]
    index_open_cells.append(id_pt)
print(index_open_cells)

#plotting the Voronoi mesh
fig = voronoi_plot_2d(vor)
plt.show()
sys.exit()

##Other way to get the cells on the boundary
#edges_from_cell = np.zeros(len(vor.points))
#for id_edge in range(len(vor.ridge_points)):
#    edges_from_cell[vor.ridge_points[id_edge][0]] += 1
#    edges_from_cell[vor.ridge_points[id_edge][1]] += 1
#print(edges_from_cell)

#print(vor.vertices.shape)
#print(vor.vertices)
#print(vor.ridge_vertices)

#Plot the cells on the boundary
fig = voronoi_plot_2d(vor)
for id_pt in bnd:
    plt.plot(vor.points[id_pt][0], vor.points[id_pt][1], 'ro')
#for vert in vor.vertices:
#    plt.plot(vert[0], vert[1], 'o', color='orange')
plt.xlim(0,2)
plt.ylim(0,2)
plt.show()
