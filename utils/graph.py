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

    def fill_graph(self): #Update here
        self.fill_cells()
        self.fill_edges()

    def fill_cells(self):
        for cid, cell in enumerate(self.voronoi):
            self.graph.add_node(cid, #id of the node
                               pos=np.array(cell["original"]),        #input point coordinates
                               area=cell["volume"])               #polygon area
            
        self.Nc = len(self.graph.nodes) #Number of cells

    def fill_edges(self):
        i = 0 #Numbering of edges. Useful for the energy
        for cid_i, cell_i in enumerate(self.voronoi):
            for face in cell_i["faces"]:
                cid_j = face["adjacent_cell"]
                if self.graph.has_edge(cid_i, cid_j):     # skip duplicates
                    continue
                elif cid_j < 0: #Boundary edge
                    self.bnd.add(cid_i) #Mark cell as being on the boundary
                    continue #See how to deal with boundary edges later on
                cid_i, cid_j = sorted((cid_i, cid_j)) #cid_i < cid_j always

                #Computing barycentre of the edge
                vidx = face["vertices"]
                verts_i = np.asarray(cell_i["vertices"])[vidx]  # shape (2,2)
                v1, v2 = verts_i 
                barycentre = .5 * (v1 + v2)
                length = np.linalg.norm(v1 - v2)
                #Computing tangent and normal
                unit_tangent = (v1 - v2) / length
                normal = self.graph.nodes[cid_j]['pos'] - self.graph.nodes[cid_i]['pos'] #normal from - towards +
                unit_normal = normal / np.linalg.norm(normal)

                self.graph.add_edge(cid_i, cid_j, normal=unit_normal, tangent=unit_tangent, bary=barycentre, length=length, id_edge=i) #add interal edge
                i += 1
        
        self.Ne = len(self.graph.edges) #Number of edges

    #def identify_bnd_cells(self):
    #    for c in range(self.Nc):
    #        region = self.voronoi.point_region[c]
    #        list_vert = self.voronoi.regions[region]
    #        if -1 in list_vert:
    #            self.bnd.add(c)

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

    #def compute_edge_quantities(self):
    #    i = 0 #Numbering for minimizing the energy
    #    for c1,c2 in self.graph.edges:
    #        verts = self.voronoi.ridge_vertices[self.graph[c1][c2]['id_ridge']]
    #        if -1 not in verts:
    #            t = self.voronoi.vertices[verts[0]] - self.voronoi.vertices[verts[1]]
    #            length = np.linalg.norm(t)
    #            assert length < np.inf #Checking no edge has infinite length
    #            self.graph[c1][c2]['length'] = length
    #            self.graph[c1][c2]['bary'] = .5 * self.voronoi.vertices[verts].sum(axis=0)
    #        else: #The force between boundary cells does not matter
    #            self.graph[c1][c2]['length'] = 1
    #            self.graph[c1][c2]['bary'] = self.voronoi.vertices[verts].sum(axis=0)              
    #        self.graph[c1][c2]['id_edge'] = i
    #        i += 1

    def bary_domain(self):
        pos_bary = np.zeros(self.d)
        for c in self.graph.nodes:
            pos_bary += self.graph.nodes[c]['pos']
        self.pos_bary = pos_bary / self.Nc
    
