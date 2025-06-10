import networkx as nx
#from scipy.spatial import Voronoi,voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import sys

class GranularMaterial:
    def __init__(self, voronoi, d, s_T=1., L=1.):
        self.d = d
        self.s_T = s_T #tresca friction
        self.L = L #Size of square domain
        self.voronoi = voronoi
        self.graph = nx.Graph()
        self.bnd = set()
        self.fill_graph()
        self.bary_domain()

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

                #Computing barycentre of the edge
                vidx = face["vertices"]
                verts_i = np.asarray(cell_i["vertices"])[vidx]  # shape (2,2)
                v1, v2 = verts_i 
                barycentre = .5 * (v1 + v2)
                length = np.linalg.norm(v1 - v2)
                #Computing tangent and normal
                unit_tangent = (v1 - v2) / length
                unit_normal = np.array((-unit_tangent[1], unit_tangent[0]))

                if cid_j < 0: #Boundary edge
                    self.bnd.add(cid_i) #Store cell as on the boundary
                    #self.graph.add_edge(cid_i, cid_j, normal=unit_normal, tangent=unit_tangent, bary=barycentre, length=length) #add boundary edge
                else: #Inner edge
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

    def plot_voronoi(self): #Need to update it
        voronoi_plot_2d(self.voronoi)
        plt.show()       

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
    
