import numpy as np
from scipy.spatial import Voronoi,voronoi_plot_2d
import matplotlib.pyplot as plt
import networkx as nx
from cvxopt import solvers,matrix,spmatrix
import sys

#Fixing seed
np.random.seed(seed=136985)

#Material parameter for friction
s = 1

#Getting the points
d = 2 #Space dimension
N = 20 #20
pts = np.random.uniform(size=d*N) * 2
points = pts.reshape((N,d))

##Test
#points = np.array([[.25, .25], [.75, .25], [.25, .75], [.75, .75], [.5, .5]])

#Getting the Voronoi mesh
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
bnd = set()
i = 0
for id_ridge in range(len(vor.ridge_points)):
    if (int(vor.ridge_points[id_ridge][0]) in to_keep) and not (int(vor.ridge_points[id_ridge][1]) in to_keep):
        bnd.add(int(vor.ridge_points[id_ridge][0]))
    elif not (int(vor.ridge_points[id_ridge][0]) in to_keep) and (int(vor.ridge_points[id_ridge][1]) in to_keep):
        bnd.add(int(vor.ridge_points[id_ridge][1]))
    elif (int(vor.ridge_points[id_ridge][0]) in to_keep) and (int(vor.ridge_points[id_ridge][1]) in to_keep):
        G.add_edge(vor.ridge_points[id_ridge][0], vor.ridge_points[id_ridge][1], id_ridge=id_ridge, id_edge=i)
        i += 1

#Computing length and barycentre of each edge
for c1,c2 in G.edges:
    verts = vor.ridge_vertices[G[c1][c2]['id_ridge']]
    length = np.linalg.norm(vor.vertices[verts[0]] - vor.vertices[verts[1]])
    assert length < np.inf #Checking no edge has infinite length
    G[c1][c2]['length'] = length
    G[c1][c2]['bary'] = .5 * vor.vertices[verts].sum(axis=0)

#Computing the area of each cell
for c1 in G.nodes:
    area = 0
    for c2 in G.neighbors(c1):
        #Compute area of each triangle
        verts = vor.ridge_vertices[G[c1][c2]['id_ridge']]
        area += .5 * np.absolute(np.cross(vor.vertices[verts[0]] - vor.vertices[verts[1]], vor.vertices[verts[0]] - vor.points[c1]))
    G.nodes[c1]['area'] = area

#Writing in graph which cells are on the boundary
for id_cell in bnd:
    G.nodes[id_cell]['bnd'] = True
inner = set(to_keep) - bnd #set of inner cells
for id_cell in inner:
    G.nodes[id_cell]['bnd'] = False

##plotting the Voronoi mesh
#fig = voronoi_plot_2d(vor)
#for id_cell in inner:
#    pts = vor.points[id_cell]
#    plt.plot(pts[0], pts[1], 'ro')
#for id_cell in bnd:
#    pts = vor.points[id_cell]
#    plt.plot(pts[0], pts[1], 'bx')
##plt.xlim(0.5,1.5)
##plt.ylim(0.5,1.5)
#plt.show()

#Computing the total bumber of inner edges
Nc = len(G.nodes) #number of cells
print('Nb cells: %i' % Nc)
Ne = len(G.edges) #number of edges and thus of inner forces
print('Nb inner edges: %i' % Ne)
#sys.exit()

#Computing the normal and tangent vector to each edge
n = np.zeros((Ne, d)) #Contains normals
t = np.zeros((Ne, d)) #Contains tangents
for (c1,c2,e) in G.edges.data('id_edge'):
    id_vert = vor.ridge_vertices[e]
    t[e] = np.array(vor.vertices[id_vert[0]] - vor.vertices[id_vert[1]])
    t[e] /= np.linalg.norm(t[e])
    n[e] = np.array((-t[e,1],t[e,0])) #at least for 2d case

#print(n)
#print(t)

#The variables for the optimization problem are the forces in the edges
#Write the energy
E = np.zeros(d*Ne)
E = matrix(E, tc='d')


#Write the inequality constraint for compression
x = n.flatten()
I = np.array((np.arange(Ne), np.arange(Ne))).T.flatten()
J = np.arange(2*Ne)
GG = spmatrix(x, I, J, tc='d') #left-hand side
h = np.zeros(Ne) #right-hand side

#Write the inequality constraint for Tresca friction law
x = np.concatenate((x, t.flatten(), -t.flatten()))
J = np.concatenate((J, J, J))
I = np.concatenate((I, I+Ne, I+2*Ne))
GG = spmatrix(x, I, J) #left-hand side
#print(G.size)
h = np.concatenate((h, s*np.ones(Ne), s*np.ones(Ne))) #rhs
#update the rhs to take into account the length of the edge
h = matrix(h, tc='d')
#print(h.size)

##Solving linear system
#sol = solvers.lp(E, GG, h)
##Vector of solution is in format f_1^x,f_1^y,f_2^x,etc...
#x = sol['x']
#print(x)
#sys.exit()

#Compute barycentre of the domain
pos_bary = np.zeros(d)
for c in G.nodes:
    pos_bary += points[c,:]
pos_bary /= Nc

#Write the equality constraints for boundary cells
force_bnd = np.zeros((d,len(bnd)))
i = 0
for c in bnd:
    G.nodes[c]['id_cell'] = i
    force_bnd[:,i] = pos_bary - vor.points[c] #vector pointing towards the barycenter
    i += 1
    
#rhs equality constraints
b = -force_bnd.T.flatten() #-
b = matrix(b, tc='d')
#print(b)

##Plotting the graph
#nx.draw(G, with_labels=True)
##plt.savefig('graph.pdf')
#plt.show()

#lhs equality constraints
A = np.zeros((d*Ne,d*len(bnd)))
for c1 in bnd:
    id_cell = G.nodes[c1]['id_cell']
    for c2 in G.neighbors(c1):
        id_edge = G[c1][c2]['id_edge']
        normal = n[id_edge]
        sign = np.dot(normal, vor.points[c2] - vor.points[c1])
        sign /= abs(sign)
        A[2*id_edge,2*id_cell] = sign #x component
        A[2*id_edge+1,2*id_cell+1] = sign #y component

A = matrix(A.T, tc='d') #Convert to sparse later on?
#print(A)

#Solving linear system
sol = solvers.lp(E, GG, h, A, b)
x = sol['x'] #Vector of solution is in format f_1^x,f_1^y,f_2^x,etc...
#print(x)

#Reformating forces
f = np.array(x).reshape((Ne, d))
#print(f)

#Checking that force balance is true on each cell
for c1 in G.nodes:
    tot = np.zeros(d)
    if G.nodes[c1]['bnd']: #bnd particle
        id_cell = G.nodes[c1]['id_cell']
        tot += force_bnd[:,id_cell] #Adding boundary force to the balance
    for c2 in G.neighbors(c1): 
        id_edge = G[c1][c2]['id_edge']
        normal = n[id_edge]
        sign = np.dot(normal, vor.points[c2] - vor.points[c1])
        sign /= abs(sign)
        tot += sign * f[id_edge]
    #print(G.nodes[c1]['bnd'])
    #print(tot)

#print(force_bnd.sum(axis=1))
#sys.exit()

#Computing the reconstruction of the stresses in each cell
for c1 in G.nodes:
    stress = np.zeros((d,d))
    for c2 in G.neighbors(c1):
        normal = n[id_edge]
        sign = np.dot(normal, vor.points[c2] - vor.points[c1])
        sign /= abs(sign)
        id_edge = G[c1][c2]['id_edge']
        stress += .5 * np.outer(sign * f[id_edge], G[c1][c2]['bary'] - vor.points[c1]) + .5 * np.outer(G[c1][c2]['bary'] - vor.points[c1], sign * f[id_edge])
    stress /= G.nodes[c1]['area']
    G.nodes[c1]['stress'] = stress

for c1 in G.nodes:
    print(G.nodes[c1]['bnd'])
    print(np.linalg.eigvalsh(G.nodes[c1]['stress']))

sys.exit()

###Plot solutions
#plotting the Voronoi mesh
fig = voronoi_plot_2d(vor)
#Plotting the facet batycentres
#plt.plot(facet_bary[:,0], facet_bary[:,1], 'ro')
#Plotting the forces
#plt.figure()
ax = plt.gca()
ax.quiver(facet_bary[:,0], facet_bary[:,1], f[:,0], f[:,1], angles='xy', scale_units='xy', scale=1, color='red')
plt.show()
