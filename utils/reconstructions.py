import numpy as np
from firedrake.petsc import PETSc
import matplotlib.pyplot as plt
import sys
from graph import *

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
        if c1 >= 0: #Actual cell. Note boundary facet.
            #Create the Mesh with DMPlex
            plex = create_plex(GM, c1)
            sys.exit()

            #Creating a list of markers and Dirichlet BC
            bnd_condition = [] #Value of normal stress on bnd
            bnd_marker = [] #Marker for the bnd

            boundary_facets = plex.getStratumIS("bnd", GM.d-1).getIndices()
            # Sort by order they appear
            for idx, facet in enumerate(boundary_facets):
                plex.setLabelValue('bnd', facet, idx) #Marking bnd

            
            for c2 in G.neighbors(c1):
                if not G[c1][c2]['bnd']:
                    id_e = G[c1][c2]['id_edge']
                    
                    #Compute outward unit normal
                    n = G.nodes[c2]['pos'] - G.nodes[c1]['pos']
                    n /= np.linalg.norm(n) #unit normal

                    #Compute BC
                    bc = np.dot(normal_stresses[:, id_e], n)
                    bnd_condition.append(bc)

                    #Marker
                    #Mark the correct bnd edge
                else: #boundary facet
                    n = G[c1][c2]['normal']
                    id_e -= GM.Ne
                    force = stress_bnd[:, id_e] * G[c1][c2]['length']
                    #force = np.zeros_like(force) #test
                    #force = -n * G[c1][c2]['length'] #test

            #Solving the system for the stress reconstruction
            stress = reconstruct_stress_polygon(plex, bnd_condition, bnd_marker)

            #Adding stress reconstruction to result list
            res.append(stress)

    return res

def create_plex(GM, cell_index):
    #Vertices of the mesh
    vertices = GM.voronoi[cell_index]['vertices'] #Vertices from Voronoi cell
    vertices.append(GM.voronoi[cell_index]['original']) #Adding vornoi center as last element

    #Cells of the mesh
    cells = []
    for f in GM.voronoi[cell_index]['faces']:
        cells.append(f['vertices'] + [len(vertices)-1])

    # === Create DMPlex mesh with topology ===
    plex = PETSc.DMPlex().createFromCellList(GM.d, cells, vertices, interpolate=True)   

    #Mark boundary facets
    plex.markBoundaryFaces("bnd")

    #Test
    print(GM.voronoi[cell_index]['faces'])
    edge_start, edge_end = plex.getDepthStratum(1)
    for edge in range(edge_start, edge_end):
        cone = plex.getCone(edge)
        print(f"Edge {edge} connects vertices {cone[:]}")

    return plex

def edge_normal(v0, v1):
    t = v1 - v0
    n = np.array([-t[1], t[0]])
    return n / np.linalg.norm(n)

def reconstruct_stress_polygon(plex, bnd_condition, bnd_marker):
    #Create mesh
    mesh = Mesh(plex)
    #Mixed FEM space
    V = VectorFunctionSpace(mesh, 'BDM', 1)
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
    L = Constant(0) * q * dx

    #Dirichlet BC
    bcs = [] #Write that

    #Solving
    res = Function(Z)
    solve(a == L, res, bcs=bcs)
    stress,lag = split(res)

    return stress
