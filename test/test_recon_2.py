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
#points = np.array([[1/6,5/6], [1/3,2/3], [5/6,2/3], [2/3,5/6], [1/6,1/3], [1/3,1/6], [2/3,1/3], [5/6,1/6]])
points = np.array([[1/4,1/4], [1/4,3/4], [3/4,1/4], [5/6,2/3], [2/3,5/6]])
#points = np.array([[1/4,1/4], [1/4,3/4], [3/4,1/4], [3/4,3/4]])

#Creating the graph
GM = GranularMaterial(points, d, s)

##plot
#GM.plot_voronoi()
#GM.plot_graph()
#sys.exit()

#Neumann condition on boundary edges
compression = 1 #1e2 #compressive force
eps = 0
S = -compression * np.array([[1, eps], [eps,1]])
stress_bnd = np.zeros((d, GM.Nbe))
for c1,c2 in GM.graph.edges:
    if GM.graph[c1][c2]['bnd']:
        id_e = GM.graph[c1][c2]['id_edge'] - GM.Ne
        normal = GM.graph[c1][c2]['normal']
        stress_bnd[:,id_e] = np.dot(S, normal)

#Assembling the system to minimize the energy
E = Energy(GM, stress_bnd)

#Test another value of inputs..
z = np.zeros((GM.Ne,3))
G = GM.graph
for c1,c2 in G.edges:
    if not G[c1][c2]['bnd']: #internal edge
        id_e = G[c1][c2]['id_edge']
        n = G[c1][c2]['normal']
        z[id_e,:2] = np.absolute(np.dot(S, n))

z = z.flatten()
en = -E.h.T * z
#print(en)
KKT = E.G.T * z + E.E
print(KKT) #Lots of non-zero here, no?
sys.exit()

#Computing the normal stresses
f = E.solve(GM)

#Stress reconstruction
stress = stress_reconstruction(GM, stress_bnd, f)
file = VTKFile('test.pvd')
for (i,s) in enumerate(stress):
    file.write(s,idx=i)

# Plot with matplotlib
fig, ax = plt.subplots()
for (i,s) in enumerate(stress):

    #Test
    mesh = s.function_space().mesh()
    scalar_space = FunctionSpace(mesh, "DG", 1)
    sigma_norm = Function(scalar_space, name="sigma_norm")
    #sigma_norm.project(sqrt(inner(s, s)))  # inner gives Frobenius inner product
    sigma_norm.interpolate(s[1,1])

    tric = tripcolor(sigma_norm, axes=ax, cmap="viridis")  # Firedrake's tripcolor wrapper
plt.colorbar(tric, ax=ax, label=r"$\sigma_{22}$") #"$\|\sigma\|_F$"
ax.set_aspect("equal")
#ax.set_title("Frobenius norm of σ")
#plt.savefig('sigma_12.png')
plt.show()
