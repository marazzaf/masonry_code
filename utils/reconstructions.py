import numpy as np
from firedrake.petsc import PETSc
from firedrake import *
import matplotlib.pyplot as plt
import sys
from graph import *
from firedrake.cython import dmcommon
from firedrake.output import VTKFile

def barycentric_coordinates_triangle(point, triangle):
    """
    Compute barycentric coordinates of a point w.r.t. a triangle in 2D.

    Parameters:
    ----------
    point : array_like
        The 2D point, e.g., [x, y].
    triangle : array_like
        List or array of three vertices [[x1, y1], [x2, y2], [x3, y3]].

    Returns:
    -------
    lambdas : numpy.ndarray
        Array of barycentric coordinates [l1, l2, l3].
    """
    A = np.array(triangle[0])
    B = np.array(triangle[1])
    C = np.array(triangle[2])
    P = np.array(point)

    # Compute areas using cross products
    def area2(u, v, w):
        return ( (v[0] - u[0]) * (w[1] - u[1])
               - (v[1] - u[1]) * (w[0] - u[0]) )

    area_total = area2(A, B, C)
    if np.isclose(area_total, 0.0):
        raise ValueError("Triangle is degenerate (zero area)")

    l1 = area2(P, B, C) / area_total
    l2 = area2(P, C, A) / area_total
    l3 = area2(P, A, B) / area_total

    return np.array([l1, l2, l3])


def stress_reconstruction(GM, stress_bnd, normal_stresses):
    res = []
    G =  GM.graph
    for c1 in G.nodes:
        if c1 >= 0: #Actual cell. Not boundary facet.
            #Create the Mesh with DMPlex
            plex = create_plex(GM, c1)

            #Creating a list of markers and Dirichlet BC
            bnd_condition = [] #Value of normal stress on bnd
            bnd_marker = [] #Marker for the bnd

            #Looping through facets of the cell
            for f in GM.voronoi[c1]['faces']:
                c2 = f['adjacent_cell']
                print(c1,c2) #PB HERE!
                if not G[c1][c2]['bnd']:
                    id_e = G[c1][c2]['id_edge'] #Id of the edge
                    
                    #Compute outward unit normal
                    n = G.nodes[c2]['pos'] - G.nodes[c1]['pos']
                    n /= np.linalg.norm(n) #unit normal

                    #Compute BC
                    stress_n, stress_t = normal_stresses[:, id_e]
                    sign = np.dot(n, G[c1][c2]['normal'])
                    t = G[c1][c2]['tangent']
                    normal_stress = stress_n * n * sign + stress_t * t
                    #print(G[c1][c2]['bary'])
                    #print(normal_stress)
                    bc = np.outer(normal_stress, n)
                    print(bc)

                else: #boundary facet
                    n = G[c1][c2]['normal']
                    id_e = GM.graph[c1][c2]['id_edge'] #To Mark edge
                    id_x = id_e - GM.Ne #To get bnd condition

                    #Compute BC
                    bc = np.outer(stress_bnd[:, id_x], n)
                    #print(bc)

                #Store BC
                bnd_condition.append(bc)

                #Find the edge in the plex
                cell_start, cell_end = plex.getDepthStratum(2)
                v1,v2 = list(np.array(f['vertices']) + cell_end)
                edge = plex.getJoin([v1, v2])
                #print(v1,v2,edge)
                
                
                #Mark the edge in the plex
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, edge, id_e) #Marking bnd
                bnd_marker.append(id_e)
            
            #Solving the system for the stress reconstruction
            stress = reconstruct_stress_polygon(plex, bnd_condition, bnd_marker)

            #Adding stress reconstruction to result list
            res.append(stress)

    return res

def create_plex(GM, cell_index):
    #Vertices of the mesh
    vertices = GM.voronoi[cell_index]['vertices'] #Vertices from Voronoi cell
    vertices.append(GM.voronoi[cell_index]['original']) #Adding voronoi center as last element

    #Cells of the mesh
    cells = []
    for f in GM.voronoi[cell_index]['faces']:
        cells.append(f['vertices'] + [len(vertices)-1])

    #Create DMPlex mesh with vertices and ells
    plex = PETSc.DMPlex().createFromCellList(GM.d, cells, vertices, interpolate=True)

    ##Mark boundary facets
    #plex.markBoundaryFaces("bnd")

    return plex

#def edge_normal(v0, v1):
#    t = v1 - v0
#    n = np.array([-t[1], t[0]])
#    return n / np.linalg.norm(n)

def reconstruct_stress_polygon(plex, bnd_condition, bnd_marker):
    #Create mesh
    mesh = Mesh(plex)

    #Mixed FEM space
    V = VectorFunctionSpace(mesh, 'RT', 1)
    W = FunctionSpace(mesh, 'DG', 0)
    Z = V * W

    #Weak form
    sigma,p = TrialFunctions(Z)
    tau,q = TestFunctions(Z)
    a = inner(div(sigma), div(tau)) * dx #LS for div equation
    mat_p = as_matrix(((0, p), (-p, 0)))
    mat_q = as_matrix(((0, q), (-q, 0)))
    a += inner(sigma, mat_q) * dx #Constraints
    a += inner(tau, mat_p) * dx #Constraints

    #Linear form
    L = Constant(0) * tau[0,0] * dx

    #Dirichlet BC
    bcs = []
    for mark,bc in zip(bnd_marker,bnd_condition):
        bc = DirichletBC(Z.sub(0), bc, mark)
        #bc = DirichletBC(V, bc, mark)
        bcs.append(bc)

    ##Test
    #test = Function(V, name='test')
    #for bc in bcs:
    #    bc.apply(test.vector())
    #file = VTKFile('bc.pvd')
    #file.write(test)
    #sys.exit()

    #Solving
    res = Function(Z, name='stress')
    params = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type' : 'mumps'}
    solve(a == L, res, bcs=bcs, solver_parameters=params)
    stress,lag = split(res)

    #print(res.at((.25,0))[0])
    #print(res.at((0,.25))[0])
    #print(res.at((.5,0))[0])
    #print(res.at((0,.5))[0])

    file = VTKFile('test.pvd')
    file.write(res.sub(0))
    sys.exit()

    return stress
