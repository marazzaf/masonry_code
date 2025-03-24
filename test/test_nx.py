import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
import networkx as nx
import sys

#Fixing seed
np.random.seed(seed=136985)

#Getting the points
N = 20 #20
pts = uniform.rvs(size=2*N) * 2
points = pts.reshape((N,2))

#Computing the Voronoi mesh
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(points)

#Creating the graph
G = nx.Graph()

#Selecting only the points that fall into [.5,1.5] x [.5,1.5]
thrash = []
for id_cell in range(len(vor.points)):
    if vor.points[id_cell][0] < .5 or vor.points[id_cell][0] > 1.5 or vor.points[id_cell][1] < .5 or vor.points[id_cell][1] > 1.5:
        thrash.append(id_cell)

#Cells to keep
to_keep = set(range(len(vor.points))) - set(thrash)
to_keep = list(to_keep)
G.add_nodes_from(to_keep) #Adding nodes to the graph

#Computing the cells on the boundary
count = 0
bnd = set()
for id_edge in range(len(vor.ridge_points)):
    if (int(vor.ridge_points[id_edge][0]) in to_keep) and not (int(vor.ridge_points[id_edge][1]) in to_keep):
        bnd.add(int(vor.ridge_points[id_edge][0]))
    elif not (int(vor.ridge_points[id_edge][0]) in to_keep) and (int(vor.ridge_points[id_edge][1]) in to_keep):
        bnd.add(int(vor.ridge_points[id_edge][1]))
    elif (int(vor.ridge_points[id_edge][0]) in to_keep) and (int(vor.ridge_points[id_edge][1]) in to_keep):
        G.add_edge(vor.ridge_points[id_edge][0], vor.ridge_points[id_edge][1], id_edge=id_edge, id_force = count)
        count += 1

#Writing in graph which cells are on the boundary
for id_cell in bnd:
    G.nodes[id_cell]['bnd'] = True
inner = set(to_keep) - bnd #set of inner cells
for id_cell in inner:
    G.nodes[id_cell]['bnd'] = False

#Test
print(G.graph)

#Plotting the graph
nx.draw(G, with_labels=True)
plt.show()
