from graph import GranularMaterial
from cvxopt import solvers,matrix,spmatrix
import numpy as np
import sys

class Energy:
    def __init__(self, GM, force_bnd): #GM is a GranularMaterial
        self.E = self.energy(GM.d, GM.Ne)
        self.equality_constraint(GM, force_bnd)
        self.inequality_constraint(GM)
    

    def energy(self, d, Ne):
        aux = np.zeros(d * Ne)
        return matrix(aux, tc='d')

    def inequality_constraint(self, GM):
        G = GM.graph #Graph of the network
        s = GM.s_T #Tresca friction parameter
        #Storing normals and tangents to each egde
        n = np.zeros((GM.Ne, GM.d))
        t = np.zeros((GM.Ne, GM.d))
        for c1,c2 in G.edges:
            id_edge = G[c1][c2]['id_edge']
            n[id_edge,:] = G[c1][c2]['normal']
            t[id_edge,:] = G[c1][c2]['tangent']
        #Write the inequality constraint for compression
        x = n.flatten()
        I = np.array((np.arange(GM.Ne), np.arange(GM.Ne))).T.flatten()
        J = np.arange(GM.d*GM.Ne)
        GG = spmatrix(x, I, J, tc='d') #left-hand side
        h = np.zeros(GM.Ne) #right-hand side

        #Write the inequality constraint for Tresca friction law
        x = np.concatenate((x, t.flatten(), -t.flatten()))
        J = np.concatenate((J, J, J))
        I = np.concatenate((I, I+GM.Ne, I+2*GM.Ne))
        self.G = spmatrix(x, I, J) #left-hand side
        #h = np.concatenate((h, s*np.ones(GM.Ne), s*np.ones(GM.Ne))) #rhs
        edge_matrix = np.zeros(GM.Ne)
        for c1,c2 in G.edges:
            id_edge = G[c1][c2]['id_edge']
            edge_matrix[id_edge] = G[c1][c2]['length']
        h = np.concatenate((h, s*edge_matrix, s*edge_matrix))
        self.h = matrix(h, tc='d')
        

    def equality_constraint(self, GM, force_bnd):
        G = GM.graph
        #rhs of the equality constraint
        self.b = -force_bnd.T.flatten() 
        self.b = matrix(self.b, tc='d')

        #lhs equality constraint
        A = np.zeros((GM.d*GM.Ne,GM.d*len(GM.bnd)))
        for c1 in GM.bnd:
            id_cell = G.nodes[c1]['id_cell']
            for c2 in G.neighbors(c1):
                id_edge = G[c1][c2]['id_edge']
                normal = G[c1][c2]['normal']
                sign = np.dot(normal, GM.voronoi.points[c2] - GM.voronoi.points[c1])
                sign /= abs(sign)
                A[2*id_edge,2*id_cell] = sign #x component
                A[2*id_edge+1,2*id_cell+1] = sign #y component
        self.A = matrix(A.T, tc='d')

    def solve(self, d, Ne):
        sol = solvers.lp(self.E, self.G, self.h, self.A, self.b, solver='glpk')
        try:
            assert sol['status'] == 'optimal'
            vec_sol = sol['x']
            return np.array(vec_sol).reshape((Ne, d))
        except AssertionError:
            print('No optimal result')
            sys.exit()
        
