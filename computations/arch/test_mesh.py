import numpy as np
from pygel3d import hmesh
import sys

# --- Hex helpers (pointy-top orientation) ---

def axial_to_xy_pointy(R, q, r):
    """
    Convert axial hex coords (q, r) to 2D Cartesian for a pointy-top hex grid.
    R = circumradius (also the edge length).
    """
    x = R * np.sqrt(3) * (q + r/2.0)
    y = R * 1.5 * r
    return float(x), float(y)

def make_hexagon(center_xy, R, orientation=np.pi/2):
    """
    Return 6 (x,y,z=0) points for a regular pointy-top hexagon centered at center_xy.
    orientation=np.pi/2 puts one vertex straight up (classic pointy-top).
    """
    cx, cy = center_xy
    pts = []
    for i in range(6):
        theta = orientation + 2*np.pi*i/6.0
        pts.append((cx + R*np.cos(theta), cy + R*np.sin(theta), 0.0))
    return pts

# --- Build a mesh of three hexes that *share edges* ---

def three_touching_hexes(R=1.0):
    """
    Create a hmesh.Manifold with three hexagons that touch edge-to-edge.
    Uses pointy-top axial neighbors (q, r) = (0,0), (1,0), (2,0).
    """
    m = hmesh.Manifold()

    # Place three neighbor centers along +q (each shares an edge with the next)
    centers_axial = [(0, 0), (1, 0), (0, 1), (0, 2)]
    centers_xy = [axial_to_xy_pointy(R, q, r) for (q, r) in centers_axial]

    for cxy in centers_xy:
        m.add_face(make_hexagon(cxy, R, orientation=np.pi/2))

    # Glue coincident boundaries so neighbors share one edge
    #hmesh.stitch(m)
    return m

#Test example
m = three_touching_hexes(R=1.0)

#Stitch all doubled vertices of mesh together
hmesh.stitch(m,rad=1e-5)
print(f"Cells: {len(list(m.faces()))}, Vertices: {len(list(m.vertices()))}")

#Loop through edges
def print_edge_to_cells(m: hmesh.Manifold):
    """
    For each undirected edge in the mesh, print:
      - its endpoint vertex ids
      - the faces (cells) on its two sides (None for boundary)
    """
    pos = m.positions()  # optional; not used but handy to have

    for h in m.halfedges():                              # iterate all halfedges
        if not m.halfedge_in_use(h):
            continue
        ho = m.opposite_halfedge(h)                      # the twin halfedge
        if h > ho:                                       # report each undirected edge once
            continue

        v0 = m.incident_vertex(h)                        # endpoint A
        v1 = m.incident_vertex(ho)                       # endpoint B (twin points opposite way)

        fL = m.incident_face(h)                          # face on one side
        fR = m.incident_face(ho)                         # face on the other side

        # Format faces; invalid/boundary sides become None
        faces = [f if m.face_in_use(f) else None for f in (fL, fR)]
        is_bdy = m.is_halfedge_at_boundary(h)

        print(f"edge ({v0}, {v1}) -> cells {faces}" + ("  [boundary]" if is_bdy else ""))

#Loop through edges of the mesh
#print_edge_to_cells(m)
#sys.exit()

#Plot
import matplotlib.pyplot as plt
pos = m.positions()
fig, ax = plt.subplots()
ax.set_aspect("equal")

# Walk each face boundary using circulate_face -> vertex ids
for f in m.faces():
    vs = list(m.circulate_face(f, mode="v"))
    xs = [pos[v][0] for v in vs] + [pos[vs[0]][0]]
    ys = [pos[v][1] for v in vs] + [pos[vs[0]][1]]
    ax.plot(xs, ys, linewidth=1.2)

ax.set_title("Three Touching Hexagons (PyGEL3D)")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.show()
