import numpy as np, pyvoro, pyvista as pv
import sys

np.random.seed(42)
N = 10
sites = np.random.rand(N, 2)              # points in [0,1]×[0,1]

cells = pyvoro.compute_2d_voronoi(
    sites.tolist(),                        # seed points
    [[0, 1], [0, 1]],                      # bounding box
    0.2                                    # block size ≈ sqrt(cell area)
)

print(cells)
sys.exit()

# ---- convert to PyVista (3-D coords) ----
points3d, faces = [], []
for cell in cells:
    pid = []
    for v in cell["vertices"]:
        try:
            idx = points3d.index(v)
        except ValueError:
            idx = len(points3d)
            points3d.append(v)
        pid.append(idx)
    faces.append([len(pid)] + pid)

pts = np.column_stack([np.array(points3d), np.zeros(len(points3d))])
mesh = pv.PolyData(pts, np.hstack(faces))

pv.Plotter().add_mesh(mesh, show_edges=True).add_points(
    np.column_stack([sites, np.zeros(len(sites))]),
    color='red', point_size=8
).show()
