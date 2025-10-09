import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.spatial import Voronoi,voronoi_plot_2d
from pygel3d import hmesh
sys.path.append('../../utils/')
from geometry import clip_voronoi_cell_vertices
from shapely.geometry import GeometryCollection

class GranularMaterial:
    def __init__(self, points, geometry: GeometryCollection, s_T = 1., d=int(2)):
        self.d = d
        self.s_T = s_T #tresca friction
        self.points = points
        self.geometry = geometry
        self.bnd = set()
        self.compute_voronoi()
        self.fill_mesh()
        #self.bary_domain()

    def compute_voronoi(self): #Voronoi mesh creation
        self.voronoi = Voronoi(self.points, qhull_options="Qbb Qc")

    def fill_mesh(self):
        #Create the empty mesh
        self.mesh = hmesh.Manifold()

        #Fill the mesh
        self.fill_inner_cells()
        self.fill_boundary_cells()

    def fill_inner_cells(self):
        self.Nc = 0 #Number of cells
        for cid, point in enumerate(self.points):
            pt_region = self.voronoi.point_region[cid]
            region = self.voronoi.regions[pt_region]
            if -1 not in region: #Closed cell (inner cell)
                vertices_id = self.voronoi.regions[pt_region]
                vertices = self.voronoi.vertices[vertices_id]
                vertices = np.column_stack([vertices[:, 0], vertices[:, 1], np.zeros_like(vertices_id)])
                self.mesh.add_face(vertices)
                self.Nc += 1


    def fill_boundary_cells(self):
        for cid, point in enumerate(self.points):
            pt_region = self.voronoi.point_region[cid]
            region = self.voronoi.regions[pt_region]
            if -1 in region:
                self.bnd.add(cid)
                print(pt_region,region)
                vertices = clip_voronoi_cell_vertices(self.voronoi, pt_region, self.geometry)[0]
                vertices = np.column_stack([vertices[:, 0], vertices[:, 1], np.zeros_like(vertices[:,0])])
                print(vertices)
                self.mesh.add_face(vertices)
                self.Nc += 1
       
        
    def plot_mesh(self):
        m = self.mesh
        pos = m.positions()
        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        # Walk each face boundary using circulate_face -> vertex ids
        for f in m.faces():
            #fig, ax = plt.subplots()
            #ax.set_aspect("equal")
            vs = list(m.circulate_face(f, mode="v"))
            print(vs)
            xs = [pos[v][0] for v in vs] + [pos[vs[0]][0]]
            ys = [pos[v][1] for v in vs] + [pos[vs[0]][1]]
            ax.plot(xs, ys, linewidth=1.2)
            plt.show()

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

    
    
