import networkx as nx
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt

class GranularMaterial:
    def __init__(self, points):
        self.voronoi = Voronoi(points)
        self.graph = nx.Graph()
        self.fill_graph()

    def fill_graph(self):
        self.fill_cells()
        self.fill_edges()

    def fill_cells(self): #Selecting only the points that fall into [.25,.75] x [.25,.75]
        thrash = []
        for id_cell in range(len(self.voronoi.points)):
            if self.voronoi.points[id_cell][0] < .25 or self.voronoi.points[id_cell][0] > .75 or self.voronoi.points[id_cell][1] < .25 or self.voronoi.points[id_cell][1] > .75:
                thrash.append(id_cell)

        #Cells to keep
        to_keep = set(range(len(self.voronoi.points))) - set(thrash)
        to_keep = list(to_keep)
        self.graph.add_nodes_from(to_keep) #Adding nodes to the graph

    def fill_edges(self):
        bnd = set() #Will keep the cells on the boundary
        i = 0
        for id_ridge in range(len(self.voronoi.ridge_points)):
            if (int(self.voronoi.ridge_points[id_ridge][0]) in self.graph.nodes) and not (int(self.voronoi.ridge_points[id_ridge][1]) in self.graph.nodes):
                bnd.add(int(self.voronoi.ridge_points[id_ridge][0]))
            elif not (int(self.voronoi.ridge_points[id_ridge][0]) in self.graph.nodes) and (int(self.voronoi.ridge_points[id_ridge][1]) in self.graph.nodes):
                bnd.add(int(self.voronoi.ridge_points[id_ridge][1]))
            elif (int(self.voronoi.ridge_points[id_ridge][0]) in self.graph.nodes) and (int(self.voronoi.ridge_points[id_ridge][1]) in self.graph.nodes):
                self.graph.add_edge(self.voronoi.ridge_points[id_ridge][0], self.voronoi.ridge_points[id_ridge][1], id_ridge=id_ridge, id_edge=i)
                i += 1
        
    def plot_graph(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()
