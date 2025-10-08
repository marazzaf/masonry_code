import numpy as np
import math
import sys
sys.path.append('../../utils/')
from energy import *
from graph_aux import *
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, GeometryCollection, MultiPolygon, Point, LineString
from shapely.ops import unary_union, linemerge
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

##plot points
#fig, ax = plt.subplots()
#ax.plot(*unit_square.exterior.xy, lw=1.5)
#for r in unit_square.interiors:
#    ax.plot(*zip(*r.coords[:]), lw=1.0)
#ax.scatter(pts[:,0], pts[:,1], s=6)
#ax.set_aspect("equal","box")
#plt.show()

#Creating Voronoi mesh
from scipy.spatial import Voronoi,voronoi_plot_2d
voro = Voronoi(pts)
##Plot
#fig = voronoi_plot_2d(voro)
#plt.show()

#Clipping Voronoi mesh to geometry
def _areal_union(g):
    """Union of polygonal pieces inside any Shapely geometry."""
    if g.is_empty:
        return None
    if isinstance(g, (Polygon, MultiPolygon)):
        return g
    if isinstance(g, GeometryCollection):
        polys = [h for h in g.geoms if isinstance(h, (Polygon, MultiPolygon))]
        if polys:
            return unary_union(polys)
        return None
    # other geometry types: no areal part
    return None

def clip_voronoi_edges_to_geometry(
    geom,
    vor: Voronoi,
    radius: float | None = None,
    use_areal_components: bool = True,
    merge_collinear: bool = False,
    eps: float = 1e-9,
):
    """
    Build Voronoi edges (as LineStrings) from a scipy.spatial.Voronoi and clip them to 'geom'.

    Args
    ----
    geom : shapely geometry (GeometryCollection/Polygon/etc.)
        The geometry to clip against. If it contains polygons, those form the clip region.
        Otherwise the geometry's envelope (expanded) is used to truncate infinite rays.
    vor : scipy.spatial.Voronoi
        2D Voronoi diagram (from your seed points).
    radius : float, optional
        Distance to extend infinite ridges before clipping. Defaults to ~4x diagram span.
    use_areal_components : bool, default True
        If True, clip to the union of polygonal parts in 'geom'. If no areal parts exist,
        falls back to an expanded envelope. If False, clip to 'geom' directly (useful if
        you want to keep segments that lie along lines in 'geom').
    merge_collinear : bool, default False
        If True, returns merged lines (linemerge over all clipped segments).
    eps : float
        Length tolerance; segments shorter than this are discarded.

    Returns
    -------
    list[LineString]  (or a MultiLineString if merge_collinear=True)
    """
    if vor.points.shape[1] != 2:
        raise ValueError("This function expects a 2D Voronoi diagram.")

    # ---- choose a clipping area to intersect against ----
    area = _areal_union(geom) if use_areal_components else None

    # a generous cap for extending rays / bounding box
    if radius is None:
        span = vor.points.ptp(axis=0).max()  # max range across x or y
        radius = max(span, 1.0) * 4.0

    # If we have areal geometry, clip to it; otherwise use an expanded envelope.
    if area is not None and not area.is_empty and area.area > 0:
        clip_target = area
        minx, miny, maxx, maxy = area.bounds
    else:
        minx, miny, maxx, maxy = geom.envelope.bounds if not geom.is_empty else (0, 0, 1, 1)
        clip_target = box(minx - radius, miny - radius, maxx + radius, maxy + radius)

    center = vor.points.mean(axis=0)

    segments = []

    # Iterate Voronoi ridges (pairs of sites) and their vertex indices
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        if v1 >= 0 and v2 >= 0:
            # Finite ridge
            a = vor.vertices[v1]
            b = vor.vertices[v2]
            seg = LineString([a, b])
        else:
            # --- FIXED: make a half-ray, not a full line ---
            t = vor.points[p2] - vor.points[p1]
            nrm = np.linalg.norm(t)
            if nrm == 0:
                continue
            t = t / nrm
            n = np.array([-t[1], t[0]])  # normal
            base = (vor.vertices[v1] if v1 >= 0 else
                    (vor.vertices[v2] if v2 >= 0 else (vor.points[p1] + vor.points[p2]) / 2.0))
            midpoint = (vor.points[p1] + vor.points[p2]) / 2.0
            direction = np.sign(np.dot(midpoint - center, n)) * n
            direction /= np.linalg.norm(direction)
            # half-ray outward only:
            a = base
            b = base + direction * radius
            seg = LineString([a, b])

        if seg.length <= eps:
            continue

        clipped = seg.intersection(clip_target)
        if clipped.is_empty:
            continue

        # Keep resulting linework
        if clipped.geom_type == "LineString":
            if clipped.length > eps:
                segments.append(clipped)
        elif clipped.geom_type == "MultiLineString":
            for s in clipped.geoms:
                if s.length > eps:
                    segments.append(s)
        # (Points may occur at tangencies; ignore.)

    if merge_collinear and segments:
        merged = linemerge(unary_union(segments))
        return merged  # MultiLineString or LineString

    return segments


# -------------------------- demo usage --------------------------
# Clip the Voronoi graph to the areal part (arch)
clipped_edges = clip_voronoi_edges_to_geometry(geometry, voro, merge_collinear=False)

# Plot
fig, ax = plt.subplots(figsize=(6, 4.5))
# arch boundary
ax.plot(*unit_square.exterior.xy, lw=1.4)
# clipped voronoi edges
for s in clipped_edges:
    x, y = s.xy
    ax.plot(x, y, lw=0.6)
# original sites
ax.plot(pts[:, 0], pts[:, 1], ".", ms=2)
ax.set_aspect("equal", "box")
ax.set_title("Voronoi edges clipped to a Shapely geometry")
plt.tight_layout()
plt.show()
