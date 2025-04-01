import networkx as nx
from scipy.spatial import Voronoi,voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np

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
        self.identify_bnd_cells()
        self.identify_bad_edges()
        self.fill_edges()
        #self.clean_graph()
        #self.compute_edge_quantities()
        #self.compute_cell_quantities()

    def fill_cells(self):
        self.graph.add_nodes_from(range(len(self.voronoi.points)))
        self.Nc = len(self.graph.nodes) #Number of cells           

    def identify_bnd_cells(self):
        self.id_bad_vertices = set()
        for c in range(self.Nc):
            region = self.voronoi.point_region[c]
            #Check if one vertex of the voronoi cell is outside the cube
            for id_vert in self.voronoi.regions[region]:
                pos_vert = self.voronoi.vertices[id_vert]
                if pos_vert[0] < 0 or pos_vert[0] > 1 or pos_vert[1] < 0 or pos_vert[1] > 1:
                    self.bnd.add(c)
                    self.id_bad_vertices.add(id_vert)

    def identify_bad_edges(self):
        self.id_bad_edges = set()
        for id_edge in range(len(self.voronoi.ridge_points)):
            list_vert = self.voronoi.ridge_vertices[id_edge]
            if len(list_vert) == 1:
                self.id_bad_edges.add(id_edge)
            elif len(set(list_vert) & self.id_bad_vertices) > 0:
                self.id_bad_edges.add(id_edge)

    def fill_edges(self):
        #Start with good edges
        for id_edge in range(len(self.voronoi.ridge_points)):
                list_points = self.voronoi.ridge_points[id_edge]
                self.graph.add_edge(list_points[0], list_points[1])
                #Add the normal to the edge
        self.Ne = len(self.graph.edges) #Number of edges

    def compute_vertices(self):
        for id_edge in range(len(self.voronoi.ridge_points)):
            if id_edge not in self.id_bad_edges:
                list_points = self.voronoi.ridge_points[id_edge]
                self.graph.add_edge(list_points[0], list_points[1])
            elif id_edge in self.id_bad_edges:
                list_vert = self.voronoi.ridge_vertices[id_edge]
                pb_vert = set(list_vert) & self.id_bad_vertices
                #for vert in pb_vert:
                    #Recompute the new vertex as an intersection
                #print(self.voronoi.vertices[list(pb_vert)])
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

def compute_intersection(d):
    res = np.zeros(d)
    return res
