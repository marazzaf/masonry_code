import numpy as np, pyvoro, pyvista as pv
import sys
from voro2pv import voro_cells_to_polydata

np.random.seed(42)
N = 50
sites = np.random.rand(N, 2)              # points in [0,1]×[0,1]

cells = pyvoro.compute_2d_voronoi(
    sites.tolist(),                        # seed points
    [[0, 1], [0, 1]],                      # bounding box
    0.1                                    # block size ≈ sqrt(cell area)
)

#print(cells)

# -----------------------------------------------------------
# Convert → PolyData and plot
# -----------------------------------------------------------
mesh = voro_cells_to_polydata(cells)

p = pv.Plotter()
p.add_mesh(mesh, show_edges=True, color="white", line_width=1)
p.add_points(np.column_stack([sites, np.zeros(len(sites))]),
             color="red", point_size=8)
p.show()
