import numpy as np
import matplotlib.pyplot as plt
import sys

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


def stress_reconstruction(GM, stress_bnd, vec_forces):
    res = []
    G =  GM.graph
    for c1 in G.nodes:
        if c1 >= 0:
            stress = np.zeros((GM.d, GM.d))
            #print(G.nodes[c1]['pos'])
            for c2 in G.neighbors(c1):
                id_e = G[c1][c2]['id_edge']
                if not G[c1][c2]['bnd']:
                    n = G.nodes[c2]['pos'] - G.nodes[c1]['pos']
                    n /= np.linalg.norm(n) #unit normal
                    t = G[c1][c2]['tangent']
                    t *= np.dot(np.array([-n[1], n[0]]), t) #Correct direction of tangent
                    force = vec_forces[id_e, 0] * n + vec_forces[id_e, 1] * t
                    #force = np.zeros_like(force) #test
                    #force = -n * G[c1][c2]['length'] #test
                else: #boundary facet
                    n = G[c1][c2]['normal']
                    id_e -= GM.Ne
                    force = stress_bnd[:, id_e] * G[c1][c2]['length']
                    #force = np.zeros_like(force) #test
                    #force = -n * G[c1][c2]['length'] #test

                #Test
                print(c2,force)

                #Adding component of stress
                coeff = abs(np.dot(G[c1][c2]['bary'] - G.nodes[c1]['pos'], n))
                stress += coeff * .5 * (np.outer(force, n) + np.outer(n, force))

            #Adding stress reconstruction to result list
            stress /= G.nodes[c1]['area']
            res.append(stress)
            print(stress)
            sys.exit()

    return res


def subdivide_from_point(vertices, point):
    """
    Subdivide a convex polygon into triangles using a given interior point.
    
    Parameters
    ----------
    vertices : (n, 2) ndarray
        Ordered polygon vertices.
    point : (2,) ndarray
        Interior point used to fan out the triangles.

    Returns
    -------
    triangles : list of (3, 2) ndarray
        List of triangle vertex arrays.
    """
    n = len(vertices)
    triangles = []

    for i in range(n):
        v0 = vertices[i]
        v1 = vertices[(i + 1) % n]
        tri = np.array([point, v0, v1])
        triangles.append(tri)

    return triangles

def compute_polygon_barycentre(vertices):
    return np.mean(vertices, axis=0)

def edge_normal(v0, v1):
    t = v1 - v0
    n = np.array([-t[1], t[0]])
    return n / np.linalg.norm(n)

def build_triangle_system(tri, n_data, t_data, internal_constraints):
    """
    Assemble the 6x6 system for a single triangle:
    - tri: (3,2) array of triangle vertices
    - n_data: dict with normal vector at edge midpoints {index: normal}
    - t_data: dict with known normal traction {index: value}
    - internal_constraints: list of (midpoint, normal, traction) from neighbors
    """
    A = []
    b = []

    def row(sigma_n, t):
        xm, ym = sigma_n['midpoint']
        nx, ny = sigma_n['normal']
        row1 = [nx, ny, 0, nx*xm, ny*xm, ny*ym]
        row2 = [0, nx, ny, nx*xm, ny*xm, ny*ym]
        return row1, row2, t[0], t[1]

    # 1. Known boundary normal stress
    edge_id = 0  # assume edge 0 is boundary (for now)
    sigma_n = {'midpoint': 0.5 * (tri[edge_id] + tri[(edge_id+1)%3]),
               'normal': n_data[edge_id]}
    row1, row2, rhs1, rhs2 = row(sigma_n, t_data[edge_id])
    A.extend([row1, row2])
    b.extend([rhs1, rhs2])

    # 2. Internal continuity constraints
    for midpoint, normal, value in internal_constraints:
        sigma_n = {'midpoint': midpoint, 'normal': normal}
        row1, row2, rhs1, rhs2 = row(sigma_n, value)
        A.append(row1)
        b.append(rhs1)

    # 3. Divergence-free constraint: div σ = 0 ⇒ ∂_x σ_{11} + ∂_y σ_{12} = 0
    #                               ⇒ ∂_x σ_{12} + ∂_y σ_{22} = 0
    # This imposes:
    # a11 + a12 = 0
    # a12 + a22 = 0
    div1 = [0, 0, 0, 1, 1, 0]  # a11 + a12 = 0
    div2 = [0, 0, 0, 0, 1, 1]  # a12 + a22 = 0
    A.append(div1)
    A.append(div2)
    b.append(0)
    b.append(0)

    return np.array(A), np.array(b)

def reconstruct_stress_polygon(vertices, tractions):
    n = len(vertices)
    center = compute_polygon_barycentre(vertices)

    triangles = []
    stress_fields = []

    for i in range(n):
        v0 = vertices[i]
        v1 = vertices[(i+1)%n]
        tri = np.array([center, v0, v1])
        triangles.append(tri)

    # Precompute edge midpoints and normals
    edge_data = []
    for i in range(n):
        p0 = vertices[i]
        p1 = vertices[(i+1)%n]
        mid = 0.5 * (p0 + p1)
        normal = edge_normal(p0, p1)
        edge_data.append({'midpoint': mid, 'normal': normal, 'traction': tractions[i]})

    # Build system for each triangle
    for i, tri in enumerate(triangles):
        # Only the boundary edge (edge 1)
        boundary_edge = 1  # (v1, v2)
        boundary_data = {
            boundary_edge: edge_data[i]['normal']
        }
        traction_data = {
            boundary_edge: edge_data[i]['traction']
        }

        # Internal edge constraints (edge 0 and 2)
        iL = (i - 1) % n
        iR = (i + 1) % n

        # Shared internal edge midpoints
        mid0 = 0.5 * (tri[0] + tri[1])
        mid2 = 0.5 * (tri[0] + tri[2])
        n0 = edge_normal(tri[1], tri[0])  # pointing outward
        n2 = edge_normal(tri[2], tri[0])  # pointing outward

        # For continuity: use average (initial guess) of left/right tractions (zero here)
        t0 = [0.0, 0.0]
        t2 = [0.0, 0.0]

        internal_constraints = [
            (mid0, n0, t0),
            (mid2, n2, t2)
        ]

        # Assemble and solve system
        M, rhs = build_triangle_system(tri, boundary_data, traction_data, internal_constraints)
        sol = np.linalg.solve(M, rhs)
        b11, b12, b22, a11, a12, a22 = sol
        B = np.array([[b11, b12], [b12, b22]])
        A = np.array([[a11, a12], [a12, a22]])
        stress_fields.append((A, B))

    return triangles, stress_fields
