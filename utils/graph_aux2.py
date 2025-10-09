import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.spatial import Voronoi,voronoi_plot_2d
from pygel3d import hmesh

class GranularMaterial:
    def __init__(self, points, d, s_T=1., L=1.):
        self.d = d
        self.s_T = s_T #tresca friction
        self.L = L #Size of square domain
        self.points = points
        #self.graph = nx.Graph()
        self.bnd = set()
        self.compute_voronoi()
        self.fill_mesh()
        #self.clip_cells() #Will clip cells to given domain
        #self.bary_domain()

    def compute_voronoi(self): #Voronoi mesh creation
        self.voronoi = Voronoi(self.points)

    def fill_mesh(self):
        #Create the empty mesh
        self.mesh = hmesh.Manifold()

        #Fill the mesh
        self.fill_inner_cells()
        #self.fill_boundary_cells()

    def fill_inner_cells(self):
        for cid, point in enumerate(self.points):
            region = self.voronoi.regions[cid]
            if -1 not in region and len(region) > 0: #Closed cell (inner cell)
                vertices_id = self.voronoi.regions[cid]
                vertices = self.voronoi.vertices[vertices_id]
                vertices = np.column_stack([vertices[:, 0], vertices[:, 1], np.zeros_like(vertices_id)])
                self.mesh.add_face(vertices)

        #self.Nc = len(self.graph.nodes) #Number of cells

    def fill_edges(self):
        G = self.graph
        voro = self.voronoi
        self.Ne = 0
        i = 0 #Numbering of edges. Useful for the energy
        for edge_id, cells in enumerate(voro.ridge_points): #only internal edges for now
            cid_i, cid_j = cells
            if G.has_edge(cid_i, cid_j):     # skip duplicates
                continue

            #Computing barycentre of the edge
            vid_1, vid_2 = voro.ridge_vertices[edge_id]
            v1, v2 = voro.vertices[vid_1], voro.vertices[vid_2]
            barycentre = .5 * (v1 + v2)
            length = np.linalg.norm(v1 - v2)

            c1,c2 = sorted((cid_i, cid_j)) #c1 < c2 always
            #Computing unit normal
            normal = G.nodes[c2]['pos'] - G.nodes[c1]['pos'] #normal from - towards + (from c1 to c2)
            unit_normal = normal / np.linalg.norm(normal)
            unit_tangent = np.array([-unit_normal[1], unit_normal[0]]) #direct rotation
            #print(i, barycentre, unit_normal)
            G.add_edge(c1, c2, normal=unit_normal, tangent=unit_tangent, bary=barycentre, length=length, id_edge=i) #add interal edge
            i += 1
            self.Ne += 1 #Increment number of internal edges

    def clip_cells(self):
        #How to do that?
        return

    # Adapted finite-polygons maker for 2D Voronoi (turns open regions into closed by capping with a big radius)
    def voronoi_finite_polygons_2d(vor: Voronoi, radius=None):
        """
        Reconstruct finite Voronoi regions from scipy.spatial.Voronoi
        Returns: (regions, vertices)
          regions: list of lists of vertex indices for each region
          vertices: array of vertex coordinates
        """
        if vor.points.shape[1] != 2:
            raise ValueError("2D input required")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max() * 4  # generous cap

        # Map point index to the ridges for that point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct each Voronoi region
        for p1, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]

            if all(v >= 0 for v in region):
                # already closed
                new_regions.append(region)
                continue

            ridges = all_ridges[p1]
            new_region = [v for v in region if v >= 0]

            for p2, v1, v2 in ridges:
                if v1 >= 0 and v2 >= 0:
                    # finite ridge
                    continue

                # Compute the missing endpoint at "infinity"
                t = vor.points[p2] - vor.points[p1]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                # midpoint of the points defining the ridge
                midpoint = (vor.points[p1] + vor.points[p2]) / 2
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v1 if v1 >= 0 else v2] + direction * radius

                new_vertices.append(far_point.tolist())
                new_region.append(len(new_vertices) - 1)

            # Order region vertices counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = [v for _, v in sorted(zip(angles, new_region))]
            new_regions.append(new_region)

            return new_regions, np.asarray(new_vertices)

#    #Check for bounary cells if they have -1 in their regions               
        
    def plot_mesh(self):
        m = self.mesh
        pos = m.positions()
        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        # Walk each face boundary using circulate_face -> vertex ids
        for f in m.faces():
            vs = list(m.circulate_face(f, mode="v"))
            xs = [pos[v][0] for v in vs] + [pos[vs[0]][0]]
            ys = [pos[v][1] for v in vs] + [pos[vs[0]][1]]
            print(xs,ys)
            ax.plot(xs, ys, linewidth=1.2)

        ax.set_title("Mesh")
        #ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.show()

    def plot_voronoi(self):
        voronoi_plot_2d(self.voronoi)
        plt.show() 

    def bary_domain(self):
        pos_bary = np.zeros(self.d)
        for c in self.graph.nodes:
            if c >= 0: #Other fake cell
                pos_bary += self.graph.nodes[c]['pos']
        self.pos_bary = pos_bary / self.Nc

    
    
