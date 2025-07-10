import numpy as np
import sys
sys.path.append('../utils')
from graph import *
from energy import *
import matplotlib.pyplot as plt

#Material parameter for friction
s = 1

#Setting points in specific positions
d = 2 #Space dimension

#Getting the points
#points = np.array([[1/6,1/6], [1/2,1/6], [5/6,1/6], [1/6,1/2], [1/2,1/2], [5/6,1/2], [1/6,5/6], [1/2,5/6], [5/6,5/6]])

#Cartesian grid example
n = 10  # divisions along each axis

# Create linspace for the grid
x = np.linspace(0, 1, n+1)
y = np.linspace(0, 1, n+1)

# Store barycenters
barycenters = []

# Loop to draw each quad and compute its barycenter
for i in range(n):
    for j in range(n):
        # Get quad corners
        x0, x1 = x[i], x[i+1]
        y0, y1 = y[j], y[j+1]
        
        quad_x = [x0, x1, x1, x0, x0]
        quad_y = [y0, y0, y1, y1, y0]
        
        # Compute barycenter
        bx = (x0 + x1) / 2
        by = (y0 + y1) / 2
        barycenters.append([bx, by])

# Convert to array
points = np.array(barycenters)

GM = GranularMaterial(points, d, s)

##Plotting points
#GM.plot_graph()
#GM.plot_voronoi()

#Neumann condition on boundary edges
compression = 1 #1 #compressive force
stress_bnd = np.zeros((d, GM.Nbe))
for c1,c2 in GM.graph.edges:
    if GM.graph[c1][c2]['bnd']:
        id_e = GM.graph[c1][c2]['id_edge'] - GM.Ne
        normal = GM.graph[c1][c2]['normal']
        stress_bnd[:,id_e] = -compression * normal
        #plt.quiver(GM.graph[c1][c2]['bary'][0], GM.graph[c1][c2]['bary'][1], stress_bnd[0,id_e], stress_bnd[1,id_e], color='red') #For plot
#plt.show()

#print(stress_bnd)
#sys.exit()

#Assembling the system to minimize the energy
E = Energy(GM, stress_bnd)
#print(E.E)
#sys.exit()

#Computing the forces
f = E.solve(GM)
print(f)
#sys.exit()

#Plotting the forces
for c1,c2 in GM.graph.edges:
    if not GM.graph[c1][c2]['bnd']:
        bary = GM.graph[c1][c2]['bary']
        id_e = GM.graph[c1][c2]['id_edge']
        n = GM.graph[c1][c2]['normal']
        #plt.quiver(bary[0], bary[1], n[0], n[1])
        t = GM.graph[c1][c2]['tangent']
        plt.quiver(bary[0], bary[1], t[0], t[1])
        #plt.quiver(bary[0], bary[1], f[id_e,0], f[id_e,1])
plt.show()
