import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import pyvista as pv
from voro2pv import voro_cells_to_polydata
import pyvoro

class GranularMaterial:
    def __init__(self, points, d, s_T=1., L=1.):
        self.d = d
        self.s_T = s_T #tresca friction
        self.L = L #Size of square domain
        self.points = points
        self.graph = nx.Graph()
        self.bnd = set()
        self.compute_voronoi()
        self.fill_graph()
        self.bary_domain()

    def compute_voronoi(self):
        size = np.sqrt(self.L**2 / len(self.points))
        #Voronoi mesh creation
        voronoi = pyvoro.compute_2d_voronoi(self.points.tolist(),                        # seed points
            [[0, self.L], [0, self.L]],                      # bounding box
            size)                                    # block size ≈ sqrt(cell area)
        self.voronoi = voronoi

    def fill_graph(self):
        self.fill_cells()
        self.fill_edges()
        #self.boundary_edge_computation()

    def fill_cells(self):
        for cid, cell in enumerate(self.voronoi):
            self.graph.add_node(cid, #id of the node
                               pos=np.array(cell["original"]),        #input point coordinates
                               area=cell["volume"])               #polygon area
            
        self.Nc = len(self.graph.nodes) #Number of cells

    def fill_edges(self):
        G = self.graph
        i = 0 #Numbering of edges. Useful for the energy
        for cid_i, cell_i in enumerate(self.voronoi):
            for face in cell_i["faces"]:
                cid_j = face["adjacent_cell"]
                print(cid_i,cid_j)
                if G.has_edge(cid_i, cid_j):     # skip duplicates
                    continue

                #Computing barycentre of the edge
                vidx = face["vertices"]
                verts_i = np.asarray(cell_i["vertices"])[vidx]  # shape (2,2)
                v1, v2 = verts_i 
                barycentre = .5 * (v1 + v2)
                length = np.linalg.norm(v1 - v2)

                c1,c2 = sorted((cid_i, cid_j)) #cid_i < cid_j always
                if c1 < 0: #Boundary edge #cid_j
                    self.bnd.add(c2) #Mark cell as being on the boundary #cid_i
                    #Test to see if cell c1 already exits!
                    if not G.has_node(c1):
                        G.add_edge(c1, c2, bary=barycentre, length=length, bnd=True) #Adding boundary edge
                    else:
                        #See what to do here!!!!
                    continue

                #Computing tangent and normal
                unit_tangent = (v1 - v2) / length
                normal = self.graph.nodes[c2]['pos'] - self.graph.nodes[c1]['pos'] #normal from - towards +
                unit_normal = normal / np.linalg.norm(normal)

                self.graph.add_edge(c1, c2, normal=unit_normal, tangent=unit_tangent, bary=barycentre, length=length, id_edge=i, bnd=False) #add interal edge
                i += 1
        
        self.Ne = len(self.graph.edges) #Number of edges

    def boundary_edge_computation(self):
        G = self.graph
        for c1,c2 in G.edges:
            c1, c2 = sorted((c1, c2))
            point = G[c1][c2]['bary']
            if G[c1][c2]['bnd']: #c1 is the 'false' cell
                triangle = [G[c2]['pos']] #Will be used to compute barycentric coordinates
                for c3 in G.neighbors(c2):
                    #Looping in neighbors of boundary cell
                    if len(triangle) < 3:
                        triangle.append(G[c3]['pos'])
                    elif len(triangle) >= 3:
                        break
                bary_coord = barycentric_coordinates_triangle(point, triangle)
                
        
    def plot_graph(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()

    def plot_voronoi(self):
        mesh = voro_cells_to_polydata(self.voronoi) #Converting voronoi mesh

        #Plot
        p = pv.Plotter()
        p.add_mesh(mesh, show_edges=True, color="white", line_width=1)
        p.add_points(np.column_stack([self.points, np.zeros(len(self.points))]),
                     color="red", point_size=8)
        p.show()     

    def bary_domain(self):
        pos_bary = np.zeros(self.d)
        for c in self.graph.nodes:
            if c >= 0: #Other fake cell
                pos_bary += self.graph.nodes[c]['pos']
        self.pos_bary = pos_bary / self.Nc

    def stress_reconstruction(self, vec_forces):
        G =  self.graph
        for c1 in G.nodes:
            if c1 in self.bnd: #No reconstruction on boundary cells
                continue
            for c2 in G.neighbors(c1):
                id_edge = G[c1][c2]['id_edge']
    
