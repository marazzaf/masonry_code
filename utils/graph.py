import networkx as nx
from scipy.spatial import Voronoi,voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import solvers,matrix

class GranularMaterial:
    def __init__(self, points, d, s_T=1., L=1.):
        self.d = d
        self.s_T = s_T #tresca friction
        self.L = L #Size of square domain
        self.voronoi = Voronoi(points)
        self.graph = nx.Graph()
        self.bnd = set()
        self.fill_graph()
        self.bary_domain()

    def fill_graph(self):
        self.fill_cells()
        self.fill_edges()
        self.identify_bnd_cells()
        self.compute_cell_quantities()
        self.compute_edge_quantities()

    def fill_cells(self):
        self.graph.add_nodes_from(range(len(self.voronoi.points)))
        self.Nc = len(self.graph.nodes) #Number of cells           

    def fill_edges(self):
        for id_ridge in range(len(self.voronoi.ridge_points)):
            list_points = self.voronoi.ridge_points[id_ridge]
            self.graph.add_edge(list_points[0], list_points[1])
            c1,c2 = list_points
            normal = self.voronoi.points[c1] - self.voronoi.points[c2]
            self.graph[c1][c2]['normal'] = normal / np.linalg.norm(normal)
            self.graph[c1][c2]['tangent'] = np.array((-self.graph[c1][c2]['normal'][1], self.graph[c1][c2]['normal'][0]))
            self.graph[c1][c2]['id_ridge'] = id_ridge
        self.Ne = len(self.graph.edges) #Number of edges

    def identify_bnd_cells(self):
        for c in range(self.Nc):
            region = self.voronoi.point_region[c]
            list_vert = self.voronoi.regions[region]
            if -1 in list_vert:
                self.bnd.add(c)

    def plot_graph(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()

    def plot_voronoi(self):
        voronoi_plot_2d(self.voronoi)
        plt.show()       

    def compute_edge_quantities(self):
        i = 0 #Numbering for minimizing the energy
        for c1,c2 in self.graph.edges:
            verts = self.voronoi.ridge_vertices[self.graph[c1][c2]['id_ridge']]
            if -1 not in verts:
                t = self.voronoi.vertices[verts[0]] - self.voronoi.vertices[verts[1]]
                length = np.linalg.norm(t)
                assert length < np.inf #Checking no edge has infinite length
                self.graph[c1][c2]['length'] = length
                self.graph[c1][c2]['bary'] = .5 * self.voronoi.vertices[verts].sum(axis=0)
            else: #The force between boundary cells does not matter
                self.graph[c1][c2]['length'] = 1
                self.graph[c1][c2]['bary'] = self.voronoi.vertices[verts].sum(axis=0)
                
            self.graph[c1][c2]['id_edge'] = i
            i += 1

    def compute_cell_quantities(self):
        for c1 in self.graph.nodes:
            area = 0
            for c2 in self.graph.neighbors(c1):
                #Compute area of each triangle
                verts = self.voronoi.ridge_vertices[self.graph[c1][c2]['id_ridge']]
                area += .5 * np.absolute(np.cross(self.voronoi.vertices[verts[0]] - self.voronoi.vertices[verts[1]], self.voronoi.vertices[verts[0]] - self.voronoi.points[c1]))
            self.graph.nodes[c1]['area'] = area

    def bary_domain(self):
        pos_bary = np.zeros(self.d)
        for c in self.graph.nodes:
            pos_bary += self.voronoi.points[c,:]
        self.pos_bary = pos_bary / self.Nc
    
