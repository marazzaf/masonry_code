import networkx as nx
from scipy.spatial import Voronoi,voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np

class GranularMaterial:
    def __init__(self, points, d, s_T, L):
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
        self.clean_graph()
        self.compute_edge_quantities()
        self.compute_cell_quantities()

    def fill_cells(self): #Selecting only the points that fall into [L/4,3*L/4] x [L/4,3*L/4]
        thrash = []
        for id_cell in range(len(self.voronoi.points)):
            if self.voronoi.points[id_cell][0] < .25*self.L or self.voronoi.points[id_cell][0] > .75*self.L or self.voronoi.points[id_cell][1] < .25*self.L or self.voronoi.points[id_cell][1] > .75*self.L:
                thrash.append(id_cell)

        #Cells to keep
        to_keep = set(range(len(self.voronoi.points))) - set(thrash)
        to_keep = list(to_keep)
        self.graph.add_nodes_from(to_keep) #Adding nodes to the graph
        #self.Nc = len(self.graph.nodes) #Number of cells

    def fill_edges(self):
        #i = 0
        for id_ridge in range(len(self.voronoi.ridge_points)):
            if (int(self.voronoi.ridge_points[id_ridge][0]) in self.graph.nodes) and not (int(self.voronoi.ridge_points[id_ridge][1]) in self.graph.nodes):
                self.bnd.add(int(self.voronoi.ridge_points[id_ridge][0]))
            elif not (int(self.voronoi.ridge_points[id_ridge][0]) in self.graph.nodes) and (int(self.voronoi.ridge_points[id_ridge][1]) in self.graph.nodes):
                self.bnd.add(int(self.voronoi.ridge_points[id_ridge][1]))
            elif (int(self.voronoi.ridge_points[id_ridge][0]) in self.graph.nodes) and (int(self.voronoi.ridge_points[id_ridge][1]) in self.graph.nodes):
                self.graph.add_edge(self.voronoi.ridge_points[id_ridge][0], self.voronoi.ridge_points[id_ridge][1], id_ridge=id_ridge) #, id_edge=i)
                #i += 1
        #self.Ne = len(self.graph.edges) #Number of edges

    def compute_edge_quantities(self):
        i = 0 #Numbering for minimizing the energy
        for c1,c2 in self.graph.edges:
            verts = self.voronoi.ridge_vertices[self.graph[c1][c2]['id_ridge']]
            t = self.voronoi.vertices[verts[0]] - self.voronoi.vertices[verts[1]]
            length = np.linalg.norm(t)
            assert length < np.inf #Checking no edge has infinite length
            self.graph[c1][c2]['length'] = length
            self.graph[c1][c2]['bary'] = .5 * self.voronoi.vertices[verts].sum(axis=0)
            self.graph[c1][c2]['tangent'] = t / length
            self.graph[c1][c2]['normal'] = np.array((-self.graph[c1][c2]['tangent'][1], self.graph[c1][c2]['tangent'][0]))
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

    def identify_bnd_cells(self):
        for c in self.graph.nodes:
            if c in self.bnd:
                self.graph.nodes[c]['bnd'] = True
            else:
                self.graph.nodes[c]['bnd'] = False

    def bary_domain(self):
        pos_bary = np.zeros(self.d)
        for c in self.graph.nodes:
            pos_bary += self.voronoi.points[c,:]
        self.pos_bary = pos_bary / self.Nc
        
    def plot_graph(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()

    def plot_voronoi(self):
        fig = voronoi_plot_2d(self.voronoi)
        for c in self.graph.nodes:
            if not self.graph.nodes[c]['bnd']:
                pts = self.voronoi.points[c]
                plt.plot(pts[0], pts[1], 'ro')
            if self.graph.nodes[c]['bnd']:
                pts = self.voronoi.points[c]
                plt.plot(pts[0], pts[1], 'bx')
        plt.show()

    def clean_graph(self): #Removes boundary cells that are only connected to boundary cells
        clean_bnd = self.bnd.copy()
        for c1 in self.bnd:
            test = True
            for c2 in self.graph.neighbors(c1):
                test = test and self.graph.nodes[c2]['bnd']
            if test:
                self.graph.remove_node(c1)
                clean_bnd.remove(c1)
        self.bnd = clean_bnd.copy()
        self.Nc = len(self.graph.nodes) #Number of cells
        self.Ne = len(self.graph.edges) #Number of edges
