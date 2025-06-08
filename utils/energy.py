from graph import GranularMaterial
from cvxopt import solvers,matrix,spmatrix
import numpy as np
import sys

class Energy:
    def __init__(self, GM, force_bnd): #GM is a GranularMaterial
        self.E = self.energy(GM, force_bnd)
        self.inequality_constraint(GM)
    

    def energy(self, GM, force_bnd):
        G = GM.graph #Graph of the network
        d = GM.d #dimension
        Nc = GM.Nc #Number of cells
        Ne = GM.Ne #Number of internal edges
        c = np.zeros(d*Nc + Ne) #Vector for energy
        print(c.shape)

        #Energy for external forces
        for c1 in GM.bnd:
            id_cell = G.nodes[c1]['id_cell']
            c[d*c1] = force_bnd[0,id_cell] #x component
            c[d*c1+1] = force_bnd[1,id_cell] #y component      

        #Energy for Tresca friction law
        s = GM.s_T #Tresca friction parameter
        edge_matrix = np.zeros(GM.Ne)
        for c1,c2 in G.edges:
            id_edge = G[c1][c2]['id_edge']
            edge_matrix[id_edge] = G[c1][c2]['length']
        c[d*Nc:] = s * edge_matrix

        return matrix(c, tc='d')

    
    def inequality_constraint(self, GM):
        G = GM.graph #Graph of the network
        d = GM.d #dimension
        Nc = GM.Nc #Number of cells
        Ne = GM.Ne #Number of internal edges
            
        #Write the inequality constraint for compression
        #Left-hand side
        GG1 = np.zeros((Ne, d*Nc+Ne))
        for c1,c2 in G.edges:
            id_edge = G[c1][c2]['id_edge']
            n = G[c1][c2]['normal']
            #sign = np.dot(normal, GM.voronoi.points[c2] - GM.voronoi.points[c1])
            #sign /= abs(sign)
            GG1[id_edge,2*c1] = -n[0] #x component
            GG1[id_edge,2*c1+1] = -n[1] #y component
            GG1[id_edge,2*c2] = n[0] #x component
            GG1[id_edge,2*c2+1] = n[1] #y component
            #Check signs above

        #Now writing the constraints for the absolute values
        GG2 = np.zeros((2*Ne, d*Nc+Ne))
        for c1,c2 in G.edges:
            id_edge = G[c1][c2]['id_edge']
            t = G[c1][c2]['tangent']
            #First inequality
            GG2[2*id_edge,2*c1] = -t[0] #x component
            GG2[2*id_edge,2*c1+1] = -t[1] #y component
            GG2[2*id_edge, d*Nc+id_edge] = -1 #for abs
            #Second inequality
            GG2[2*id_edge+1,2*c2] = t[0] #x component
            GG2[2*id_edge+1,2*c2+1] = t[1] #y component
            GG2[2*id_edge+1, d*Nc+id_edge] = -1 #for abs

        GG = np.concatenate((GG1, GG2))
        self.G = matrix(GG, tc='d')

        #Right-hand side
        h = np.zeros(3*Ne)
        self.h = matrix(h, tc='d')
        

#    def equality_constraint(self, GM, force_bnd):
#        G = GM.graph
#        #rhs of the equality constraint
#        self.b = -force_bnd.T.flatten() 
#        self.b = matrix(self.b, tc='d')
#
#        #lhs equality constraint
#        A = np.zeros((GM.d*GM.Ne,GM.d*len(GM.bnd)))
#        for c1 in GM.bnd:
#            id_cell = G.nodes[c1]['id_cell']
#            for c2 in G.neighbors(c1):
#                id_edge = G[c1][c2]['id_edge']
#                normal = G[c1][c2]['normal']
#                sign = np.dot(normal, GM.voronoi.points[c2] - GM.voronoi.points[c1])
#                sign /= abs(sign)
#                A[2*id_edge,2*id_cell] = sign #x component
#                A[2*id_edge+1,2*id_cell+1] = sign #y component
#        self.A = matrix(A.T, tc='d')

    def solve(self, d, Nc):
        sol = solvers.lp(self.E, self.G, self.h) #, solver='glpk')
        try:
            assert sol['status'] == 'optimal'
            vec_sol = sol['x']
            return np.array(vec_sol).reshape((Nc, d))
        except AssertionError:
            print('No optimal result')
            print(sol['primal infeasibility'])
            sys.exit()
        
