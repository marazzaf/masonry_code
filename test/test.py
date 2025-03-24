import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
import sys

#Fixing seed
np.random.seed(seed=136985)

#Getting the points
N = 20 #20
pts = uniform.rvs(size=2*N) * 2
points = pts.reshape((N,2))

#Test
#points = np.array([[.45, .45], [1.55, .45], [.45, 1.55], [1.55, 1.55], [.5, .5]])

#Computing the Voronoi mesh
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(points)

#Selecting only the points that fall into [.5,1.5] x [.5,1.5]
thrash = []
for id_cell in range(len(vor.points)):
    if vor.points[id_cell][0] < .5 or vor.points[id_cell][0] > 1.5 or vor.points[id_cell][1] < .5 or vor.points[id_cell][1] > 1.5:
        thrash.append(id_cell)

#Cells to keep
to_keep = set(range(len(vor.points))) - set(thrash)
print(to_keep)
to_keep = list(to_keep)

#Computing the cells on the boundary
bnd = set()
for id_edge in range(len(vor.ridge_points)):
    if (int(vor.ridge_points[id_edge][0]) in to_keep) and not (int(vor.ridge_points[id_edge][1]) in to_keep):
        bnd.add(int(vor.ridge_points[id_edge][0]))
    elif not (int(vor.ridge_points[id_edge][0]) in to_keep) and (int(vor.ridge_points[id_edge][1]) in to_keep):
        bnd.add(int(vor.ridge_points[id_edge][1]))

print(bnd)

inner = set(to_keep) - bnd #set of inner cells

import networkx as nx
G = nx.Graph()
G.add_nodes_from(to_keep)
nx.draw(G, with_labels=True)
plt.show()
sys.exit()

#plotting the Voronoi mesh
fig = voronoi_plot_2d(vor)
for id_cell in inner:
    pts = vor.points[id_cell]
    plt.plot(pts[0], pts[1], 'ro')
for id_cell in bnd:
    pts = vor.points[id_cell]
    plt.plot(pts[0], pts[1], 'bx')
#plt.xlim(0.5,1.5)
#plt.ylim(0.5,1.5)
plt.show()
