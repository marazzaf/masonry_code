# pip install shapely==2.* scipy numpy
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box
from shapely.ops import unary_union
from shapely.geometry.polygon import orient

# -------- helpers --------

def _areal_union(g):
    """Union of polygonal parts inside any Shapely geometry (or None if none)."""
    if g.is_empty:
        return None
    if isinstance(g, (Polygon, MultiPolygon)):
        return g
    if isinstance(g, GeometryCollection):
        polys = [h for h in g.geoms if isinstance(h, (Polygon, MultiPolygon))]
        return unary_union(polys) if polys else None
    return None

def _finite_cell_polygon(vor: Voronoi, i: int, radius: float | None = None) -> Polygon:
    """
    Return a finite polygon for Voronoi cell i.
    If the region is infinite, extend open ridges out to 'radius' and close it.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Only 2D Voronoi supported.")
    if radius is None:
        radius = vor.points.ptp(axis=0).max() * 4.0

    center = vor.points.mean(axis=0)
    region_index = vor.point_region[i]
    region = vor.regions[region_index]

    # gather ridges touching i
    ridges = []
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        if p1 == i or p2 == i:
            ridges.append((p2 if p1 == i else p1, v1, v2))

    if region and all(v >= 0 for v in region):
        vs = np.asarray(vor.vertices)[region]
        return Polygon(vs).buffer(0)

    # reconstruct infinite region
    new_vertices = vor.vertices.tolist()
    new_region = [v for v in (region or []) if v >= 0]

    for p2, v1, v2 in ridges:
        if v1 >= 0 and v2 >= 0:
            continue
        t = vor.points[p2] - vor.points[i]
        nrm = np.linalg.norm(t)
        if nrm == 0:
            continue
        t = t / nrm
        n = np.array([-t[1], t[0]])

        base = (vor.vertices[v1] if v1 >= 0 else
                (vor.vertices[v2] if v2 >= 0 else (vor.points[i] + vor.points[p2]) / 2.0))
        midpoint = (vor.points[i] + vor.points[p2]) / 2.0
        direction = np.sign(np.dot(midpoint - center, n)) * n
        direction /= np.linalg.norm(direction)

        far_pt = base + direction * radius
        new_vertices.append(far_pt.tolist())
        new_region.append(len(new_vertices) - 1)

    vs = np.asarray([new_vertices[v] for v in new_region], dtype=float)
    if len(vs) < 3:
        return Polygon()
    c = vs.mean(axis=0)
    ang = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
    vs = vs[np.argsort(ang)]
    return Polygon(vs).buffer(0)

# -------- main --------

def clip_voronoi_cell_vertices(
    vor: Voronoi,
    cell_index: int,
    geom,
    *,
    use_areal_components: bool = True,
    envelope_padding: float = 0.1,
    closed: bool = False,
    dtype=float,
):
    """
    Return the clipped Voronoi cell (for 'cell_index') as a list of 2D NumPy arrays.
    Each array contains the exterior-ring vertices of a polygon piece (no holes).

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
    cell_index : int
    geom : Shapely geometry (GeometryCollection/Polygon/MultiPolygon/…)
    use_areal_components : bool
        If True (default), intersect with union of polygonal parts in 'geom'.
        If none exist, returns [].
        If False, intersect with 'geom' directly (may still yield no areal parts → []).
    envelope_padding : float
        Sets the radius used to cap infinite cells (fraction of bbox span).
    closed : bool
        If True, repeat the first vertex at the end of each ring.
    dtype : numpy dtype for output coordinates.

    Returns
    -------
    list[np.ndarray]  # each shaped (N, 2), exterior-only
    """
    if cell_index < 0 or cell_index >= vor.points.shape[0]:
        raise IndexError("cell_index out of range.")

    # radius from geom bounds (preferred) or sites bbox (fallback)
    area = _areal_union(geom) if use_areal_components else None
    if area is not None and not area.is_empty and area.area > 0:
        minx, miny, maxx, maxy = area.bounds
    else:
        minx, miny = vor.points.min(axis=0)
        maxx, maxy = vor.points.max(axis=0)

    span = max(maxx - minx, maxy - miny)
    radius = max(span * (1.0 + 2.0 * envelope_padding), 1.0)

    cell_poly = _finite_cell_polygon(vor, cell_index, radius=radius)
    clip_target = area if use_areal_components else geom
    if clip_target is None or clip_target.is_empty:
        return []

    clipped = cell_poly.intersection(clip_target).buffer(0)
    if clipped.is_empty:
        return []

    polys = [clipped] if isinstance(clipped, Polygon) else [
        p for p in clipped.geoms if isinstance(p, Polygon)
    ]
    if not polys:
        return []

    out = []
    for poly in polys:
        poly = orient(poly, sign=1.0)  # CCW exterior
        coords = list(poly.exterior.coords)
        if not closed and len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        arr = np.asarray(coords, dtype=dtype)[:, :2]  # (N, 2)
        out.append(arr)
    return out
