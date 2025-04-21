import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
import sys

#Fixing seed
np.random.seed(seed=136985)

#Space parameters
d = 2 #Space dimension
N = 503 #20

#Getting the points
#Points inside a unit square
pts = np.random.uniform(size=d*N)
points = pts.reshape((N,d))

#Points inside a unit ball
N = 5 #20
theta = uniform.rvs(size=N) * np.pi * 2
r = uniform.rvs(size=N)
#points = np.array(r*[np.cos(theta), np.sin(theta)]).T

##Grid
##points = np.array([[.45, .45], [1.55, .45], [.45, 1.55], [1.55, 1.55], [.5, .5]])
#aux = np.linspace(0, 1, N, endpoint=True)
#from itertools import product
#points = list(product(aux, aux))

#Computing the Voronoi mesh
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(points)

#plotting the Voronoi mesh
fig = voronoi_plot_2d(vor)
#plt.xlim(0,1)
#plt.ylim(0,1)
#plt.xlim(-5,5)
#plt.ylim(-5,5)
plt.show()

sys.exit()

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
