import numpy as np
import sys
sys.path.append('../utils')
from graph import *
from energy import *
import matplotlib.pyplot as plt
from reconstructions import stress_reconstruction

#Material parameter for friction
s = 10

#Space dimension
d = 2 

#Getting the points
#points = np.array([[1/6,5/6], [1/3,2/3], [5/6,2/3], [2/3,5/6], [1/6,1/3], [1/3,1/6], [2/3,1/3], [5/6,1/6]]) #Not working
#points = np.array([[1/4,1/4], [1/4,3/4], [3/4,1/4], [5/6,2/3], [2/3,5/6]]) #Interesting test
#points = np.array([[1/4,1/4], [1/4,3/4], [3/4,1/4], [3/4,3/4]]) #First verification test
points = np.array([[1/2,1/6], [1/2,5/6], [1/6,1/2], [5/6,1/2]]) #Second verification test
#points = np.array([[1/4,1/4], [1/4,3/4], [3/4,1/4], [3/4,3/4], [1/2,1/2]])
#points = np.array([[1/6,1/6], [1/6,5/6], [5/6,1/6], [5/6,5/6], [1/2,1/2]])

#Test
nx, ny = 6, 6
x = np.linspace(0.0, 1.0, nx+1)
y = np.linspace(0.0, 1.0, ny+1)
X, Y = np.meshgrid(x, y, indexing='xy')          # (ny+1, nx+1)
nodes = np.stack((X, Y), axis=-1)                # (ny+1, nx+1, 2)
xc = 0.5 * (x[:-1] + x[1:])
yc = 0.5 * (y[:-1] + y[1:])
points = []
for xx in xc:
    for yy in yc:
        points.append([xx, yy])
points = np.array(points)

#Creating the graph
GM = GranularMaterial(points, d, s)

##plot
GM.plot_voronoi()
#GM.plot_graph()
#sys.exit()

#Neumann condition on boundary edges
compression = 1 #compressive force
eps = 1 #1 #.5
S = -compression * np.array([[1, eps], [eps,1]])
stress_bnd = np.zeros((d, GM.Nbe))
for c1,c2 in GM.graph.edges:
    if GM.graph[c1][c2]['bnd']:
        id_e = GM.graph[c1][c2]['id_edge'] - GM.Ne
        normal = GM.graph[c1][c2]['normal']
        stress_bnd[:,id_e] = np.dot(S, normal)

#Assembling the system to minimize the energy
E = Energy(GM, stress_bnd)

#Computing the normal stresses
f = E.solve(GM)
print(f)

#Stress reconstruction
stress = stress_reconstruction(GM, stress_bnd, f)
file = VTKFile('test.pvd')
for (i,s) in enumerate(stress):
    file.write(s,idx=i)
#sys.exit()

# Plot with matplotlib
fig, ax = plt.subplots()
for (i,s) in enumerate(stress):

    #Test
    mesh = s.function_space().mesh()
    scalar_space = FunctionSpace(mesh, "DG", 1)
    sigma_norm = Function(scalar_space, name="sigma_norm")
    #sigma_norm.project(sqrt(inner(s, s)))  # inner gives Frobenius inner product
    sigma_norm.interpolate(s[1,1])

    tric = tripcolor(sigma_norm, axes=ax, cmap="jet")  # Firedrake's tripcolor wrapper
plt.colorbar(tric, ax=ax, label=r"$\sigma_{22}$") #"$\|\sigma\|_F$"
ax.set_aspect("equal")
#plt.savefig('sigma_12.png')
plt.show()
