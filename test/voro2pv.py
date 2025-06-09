##############################################################################
# voro2pv.py ── Convert 2-D pyvoro output → PyVista PolyData
##############################################################################
import numpy as np
import pyvista as pv


def voro_cells_to_polydata(cells, *, zlevel=0.0):
    """
    Convert the list returned by ``pyvoro.compute_2d_voronoi`` to
    a ``pyvista.PolyData`` mesh.

    Parameters
    ----------
    cells : list[dict]
        The list of cell dictionaries coming from pyvoro. Each dict contains
        (among others) the key ``"vertices"`` with a list of cell vertices.
        The order is already suitable for building polygon faces.
    zlevel : float, default 0.0
        The z-coordinate assigned to every vertex so that the 2-D diagram
        becomes a 3-D, z-flat surface (PyVista expects 3-D points).

    Returns
    -------
    pv.PolyData
        A polygonal mesh whose faces are the Voronoi cells.
    """
    # ------------------------------------------------------------------
    # 1.  Build a global unique-vertex table  (dict: coordinate -> index)
    # ------------------------------------------------------------------
    vertex_id = {}                 # maps 2-D coord tuple → global index
    vertices2d = []                # list of unique 2-D points

    def get_vid(coord):
        key = tuple(coord)
        if key not in vertex_id:
            vertex_id[key] = len(vertices2d)
            vertices2d.append(coord)
        return vertex_id[key]

    # ------------------------------------------------------------------
    # 2.  Build the vtk "faces" array
    #     [n₀, i₀¹, i₀², …, n₁, i₁¹, …]  (same as VTK_POLY_DATA)
    # ------------------------------------------------------------------
    face_stream = []
    for cell in cells:
        verts = cell["vertices"]
        ids = [get_vid(v) for v in verts]
        face_stream.extend([len(ids), *ids])

    faces = np.array(face_stream, dtype=np.int64)

    # ------------------------------------------------------------------
    # 3.  Promote 2-D vertices to 3-D
    # ------------------------------------------------------------------
    points3d = np.column_stack([np.asarray(vertices2d), 
                                np.full(len(vertices2d), zlevel)])

    # ------------------------------------------------------------------
    # 4.  Construct and return PolyData
    # ------------------------------------------------------------------
    return pv.PolyData(points3d, faces)
