import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import pyvista as pv
from voro2pv import voro_cells_to_polydata
import pyvoro
from reconstructions import *
from itertools import combinations

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
        self.boundary_edge_computation()

    def fill_cells(self):
        for cid, cell in enumerate(self.voronoi):
            face_dict = dict()
            for i in cell['faces']:
                face_dict[i['adjacent_cell']] = i['vertices']
            self.graph.add_node(cid, #id of the node
                               pos=np.array(cell["original"]),        #input point coordinates
                                area=cell["volume"],               #polygon area
                                face_dict=face_dict) #To store vertices of adjacent facets
            
        self.Nc = len(self.graph.nodes) #Number of cells

    def fill_edges(self):
        G = self.graph
        self.Ne = 0
        i = 0 #Numbering of edges. Useful for the energy
        for cid_i, cell_i in enumerate(self.voronoi):
            for face in cell_i["faces"]:
                cid_j = face["adjacent_cell"]
                if G.has_edge(cid_i, cid_j):     # skip duplicates
                    continue

                #Computing barycentre of the edge
                vidx = face["vertices"]
                verts_i = np.asarray(cell_i["vertices"])[vidx]  # shape (2,2)
                v1, v2 = verts_i 
                barycentre = .5 * (v1 + v2)
                #Compute unit tangent
                length = np.linalg.norm(v1 - v2)
                unit_tangent = (v1 - v2) / length

                c1,c2 = sorted((cid_i, cid_j)) #c1 < c2 always
                if c1 < 0: #Boundary edge
                    self.bnd.add(c2) #Mark cell as being on the boundary

                    #Computing unit normal
                    normal = np.array([-unit_tangent[1], unit_tangent[0]])
                    opposite_direction = G.nodes[c2]['pos'] - barycentre
                    opposite_direction /= np.linalg.norm(opposite_direction)
                    normal *= -np.dot(normal, opposite_direction)
                    
                    if not G.has_node(c1): #Test to see if cell c1 already exits!
                        G.add_edge(c1, c2, bary=barycentre, length=length, normal=normal, bnd=True) #Adding boundary edge
                    else:
                        c1p = -self.Nc + c1
                        G.add_edge(c1p, c2, bary=barycentre, length=length, normal=normal, bnd=True) #Adding boundary edge
                        #verts = G.nodes[c2]['face_dict'][c1]
                        G.nodes[c2]['face_dict'][c1p] = G.nodes[c2]['face_dict'].pop(c1) #Modify dict in cell 2
                else: #internal edge
                    #Computing unit normal
                    normal = G.nodes[c2]['pos'] - G.nodes[c1]['pos'] #normal from - towards +
                    unit_normal = normal / np.linalg.norm(normal)
                    unit_tangent = np.array([-unit_normal[1], unit_normal[0]])
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
                triangle_id = [c2]
                triangle_coord = [G.nodes[c2]['pos']] #Will be used to compute barycentric coordinates
                for c3 in G.neighbors(c2):
                    #Looping in neighbors of boundary cell
                    if c3 >= 0:
                    #if len(triangle_id) < 3 and c3 >= 0:
                        triangle_id.append(c3)
                        triangle_coord.append(G.nodes[c3]['pos'])
                    #elif len(triangle_id) >= 3:
                    #    break
                #Compute barycentric coordinates in the triangle
                for sub_tri_id,sub_tri_coord in zip(combinations(triangle_id,3), combinations(triangle_coord,3)):
                    try:
                        bary_coord = barycentric_coordinates_triangle(point, sub_tri_coord)
                        G[c1][c2]['bary_coord'] = bary_coord
                        G[c1][c2]['bary_points'] = sub_tri_id
                        G[c1][c2]['id_edge'] = self.Ne + i #Used for stress boundary conditions
                        i += 1
                        break
                    except ValueError:
                        print('Pb with barycentric coordinate computation!')
                        sys.exit()
                    
                
        
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

    
    
