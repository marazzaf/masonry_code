import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.spatial import Voronoi,voronoi_plot_2d

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
        self.clip_cells() #Will clip cells to given domain
        #self.bary_domain()

    def compute_voronoi(self): #Voronoi mesh creation
        self.voronoi = Voronoi(self.points)

    def fill_graph(self):
        self.fill_cells()
        self.fill_edges()
        self.boundary_edge_computation()

    def fill_cells(self):
        for cid, point in enumerate(self.voronoi.points):
            self.graph.add_node(cid, pos=point)

        self.Nc = len(self.graph.nodes) #Number of cells

    def fill_edges(self):
        G = self.graph
        voro = self.voronoi
        self.Ne = 0
        i = 0 #Numbering of edges. Useful for the energy
        for edge_id, cells in enumerate(voro.ridge_points):
            print(edge_id,cells)
            cid_i, cid_j = cells
            if G.has_edge(cid_i, cid_j):     # skip duplicates
                continue

            #Computing barycentre of the edge
            vid_1, vid_2 = voro.ridge_vertices[edge_id]
            print(vid_1,vid_2)
            v1, v2 = voro.vertices[vid_1], voro.vertices[vid_2]
            barycentre = .5 * (v1 + v2)
            length = np.linalg.norm(v1 - v2)
            sys.exit()

            c1,c2 = sorted((cid_i, cid_j)) #c1 < c2 always
            if c1 < 0: #Boundary edge
                self.bnd.add(c2) #Mark cell as being on the boundary
                #Computing unit normal
                tangent = v1 - v2
                n = barycentre - G.nodes[c2]['pos']
                normal = np.array([-tangent[1], tangent[0]])
                unit_normal = normal / np.linalg.norm(normal)
                unit_normal *= np.sign(np.dot(unit_normal, n)) #outwards unit normal
                    
                if not G.has_node(c1): #Test to see if cell c1 already exits!
                    G.add_edge(c1, c2, bary=barycentre, length=length, normal=unit_normal, bnd=True) #Adding boundary edge
                else:
                    c1p = -self.Nc + c1
                    G.add_edge(c1p, c2, bary=barycentre, length=length, normal=unit_normal, bnd=True) #Adding boundary edge
                    G.nodes[c2]['face_dict'][c1p] = G.nodes[c2]['face_dict'].pop(c1) #Modify dict in cell 2
                        
            else: #internal edge
                #Computing unit normal
                normal = G.nodes[c2]['pos'] - G.nodes[c1]['pos'] #normal from - towards + (from c1 to c2)
                unit_normal = normal / np.linalg.norm(normal)
                unit_tangent = np.array([-unit_normal[1], unit_normal[0]])
                #print(i, barycentre, unit_normal)
                G.add_edge(c1, c2, normal=unit_normal, tangent=unit_tangent, bary=barycentre, length=length, id_edge=i, bnd=False) #add interal edge
                i += 1
                self.Ne += 1 #Increment number of internal edges

    def boundary_edge_computation(self):
        G = self.graph
        self.Nbe = len(G.edges) - self.Ne #number of boundary edges
        i = 0
        for cc1,cc2 in G.edges:
            c1, c2 = sorted((cc1, cc2))
            point = G[c1][c2]['bary']
            if G[c1][c2]['bnd']: #c1 is the 'false' cell
                G[c1][c2]['bary_coord'] = np.array([1])
                G[c1][c2]['bary_points'] = [c2]
                G[c1][c2]['id_edge'] = self.Ne + i #Used for stress boundary conditions
                i += 1
        
    def plot_graph(self):
        nx.draw(self.graph, with_labels=True)
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

    
    
