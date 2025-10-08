import numpy as np
import math
import sys
sys.path.append('../../utils/')
from energy import *
from graph_aux import *
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, GeometryCollection, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep

#Fixing seed
np.random.seed(seed=136985)

#Material parameter for friction
s = 10
d = 2 #Space dimension

#Create shapely geometry of a unit square
unit_square = box(0.0, 0.0, 1.0, 1.0)
geometry = GeometryCollection([unit_square])

##test
#print(unit_square.area)      # 1.0
#print(unit_square.bounds)    # (0.0, 0.0, 1.0, 1.0)
#print(unit_square.is_valid)  # True

#test
def _areal_union(g):
    """Union of polygonal pieces inside any Shapely geometry."""
    if g.is_empty:
        return None
    if isinstance(g, (Polygon, MultiPolygon)):
        return g.buffer(0)
    if isinstance(g, GeometryCollection):
        polys = [h for h in g.geoms if isinstance(h, (Polygon, MultiPolygon))]
        return unary_union(polys).buffer(0) if polys else None
    return None

def _rotate(points_xy, angle_rad, center):
    if not angle_rad:
        return points_xy
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    cx, cy = center
    X = points_xy - np.array([[cx, cy]])
    R = np.array([[c, -s],[s, c]])
    return (X @ R.T) + np.array([[cx, cy]])

def equidistant_points_in_geometry(
    geom,
    spacing: float,
    lattice: str = "hex",      # "hex" (triangular) or "square"
    include_boundary: bool = True,
    rotate_degrees: float = 0.0,
    return_points: bool = False
):
    """
    Generate equidistant points *inside* the polygonal parts of 'geom'.

    Parameters
    ----------
    geom : shapely geometry (GeometryCollection, Polygon, MultiPolygon, etc.)
    spacing : float
        Nearest-neighbor spacing (grid constant).
    lattice : {"hex","square"}
        'hex' yields a triangular lattice (best isotropy). 'square' yields orthogonal grid.
    include_boundary : bool
        If True, keep points on the boundary (covers); else require strict interior (contains).
    rotate_degrees : float
        Optional rotation of the lattice about the domain center.
    return_points : bool
        If True, return list[shapely.geometry.Point]; else return np.ndarray (N,2).

    Returns
    -------
    np.ndarray shape (N,2) (default) or list[Point]
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive.")

    area = _areal_union(geom)
    if area is None or area.is_empty or area.area <= 0:
        return [] if return_points else np.empty((0, 2))

    minx, miny, maxx, maxy = area.bounds
    cx, cy = 0.5*(minx+maxx), 0.5*(miny+maxy)

    # Build an axis-aligned lattice covering the bbox (pad a bit for rotation)
    pad = spacing
    minx -= pad; miny -= pad; maxx += pad; maxy += pad

    if lattice.lower() == "hex":
        dx = spacing
        dy = spacing * math.sqrt(3)/2.0   # vertical pitch
        n_rows = int(math.ceil((maxy - miny)/dy)) + 1
        n_cols = int(math.ceil((maxx - minx)/dx)) + 2

        coords = []
        for r in range(n_rows):
            y = miny + r*dy
            x0 = minx + (0.5*dx if (r % 2) else 0.0)  # staggered rows
            for c in range(n_cols):
                x = x0 + c*dx
                coords.append((x, y))
        P = np.asarray(coords, dtype=float)
    elif lattice.lower() == "square":
        nx = int(math.ceil((maxx - minx)/spacing)) + 1
        ny = int(math.ceil((maxy - miny)/spacing)) + 1
        xs = minx + spacing*np.arange(nx)
        ys = miny + spacing*np.arange(ny)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        P = np.column_stack([X.ravel(), Y.ravel()])
    else:
        raise ValueError("lattice must be 'hex' or 'square'.")

    # Optional rotation about the domain center
    angle = math.radians(rotate_degrees)
    if abs(angle) > 1e-14:
        P = _rotate(P, angle, (cx, cy))

    # Keep only points inside the area (boundary optional)
    A = prep(area)
    keep = []
    if include_boundary:
        for x, y in P:
            if A.covers(Point(x, y)):
                keep.append((x, y))
    else:
        for x, y in P:
            if area.contains(Point(x, y)):
                keep.append((x, y))

    if return_points:
        return [Point(x, y) for x, y in keep]
    return np.asarray(keep, dtype=float)


#Putting the points within the geometry
pts = equidistant_points_in_geometry(geometry, spacing=0.08, lattice="hex", rotate_degrees=15)

#plot points
fig, ax = plt.subplots()
ax.plot(*unit_square.exterior.xy, lw=1.5)
for r in unit_square.interiors:
    ax.plot(*zip(*r.coords[:]), lw=1.0)
ax.scatter(pts[:,0], pts[:,1], s=6)
ax.set_aspect("equal","box"); plt.show()

sys.exit()

#Creating the graph
GM = GranularMaterial(points, d, s)

#Plotting points
GM.plot_graph()
GM.plot_voronoi()
sys.exit()

#Neumann condition on boundary edges
compression = 1 #1e2 #compressive force
eps = 1 #1 #.1
stress_bnd = np.zeros((d, GM.Nbe))
for c1,c2 in GM.graph.edges:
    if GM.graph[c1][c2]['bnd']:
        id_e = GM.graph[c1][c2]['id_edge'] - GM.Ne
        normal = GM.graph[c1][c2]['normal']
        bary = GM.graph[c1][c2]['bary']
        if bary[1] > .9 and (bary[0] - .5) < .1:
            stress_bnd[:,id_e] = -compression * normal
        else:
            stress_bnd[:,id_e] = -eps * normal

#Assembling the system to minimize the energy
E = Energy(GM, stress_bnd)

#Computing the normal stresses
f = E.solve(GM)
#print(f)

#Stress reconstruction
stress = stress_reconstruction(GM, stress_bnd, f)
file = VTKFile('sol.pvd')
# Plot with matplotlib
fig, ax = plt.subplots()
for (i,s) in enumerate(stress):
    file.write(s,idx=i)

    #Test
    mesh = s.function_space().mesh()
    scalar_space = FunctionSpace(mesh, "DG", 1)
    sigma_norm = Function(scalar_space, name="sigma_norm")
    #sigma_norm.project(sqrt(inner(s, s)))  # inner gives Frobenius inner product
    sigma_norm.interpolate(s[1,0])

    tric = tripcolor(sigma_norm, axes=ax)#, cmap="jet")  # Firedrake's tripcolor wrapper
plt.colorbar(tric, ax=ax, label=r"$\sigma_{21}$")
#plt.colorbar(tric, ax=ax, label=r"$\|\sigma\|_F$")
ax.set_aspect("equal")
#ax.set_title(r"Frobenius norm of $\sigma$")
plt.savefig('hom_21.png')
plt.show()
