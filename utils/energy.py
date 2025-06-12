from graph import GranularMaterial
from cvxopt import solvers,matrix,spmatrix
import numpy as np
import sys

class Energy:
    def __init__(self, GM, force_bnd): #GM is a GranularMaterial
        self.E = self.energy(GM, force_bnd)
        self.inequality_constraint(GM)
        self.equality_constraint(GM)
    

    def energy(self, GM, force_bnd):
        G = GM.graph #Graph of the network
        d = GM.d #dimension
        Nc = GM.Nc #Number of cells
        Ne = GM.Ne #Number of internal edges
        c = np.zeros(d*Nc + Ne) #Vector for energy

        #Energy for external forces
        for c1 in GM.bnd:
            id_cell = G.nodes[c1]['id_cell']
            c[d*c1] = -force_bnd[0,id_cell] #x component
            c[d*c1+1] = -force_bnd[1,id_cell] #y component

        #Energy for Tresca friction law
        s = GM.s_T #Tresca friction parameter
        edge_matrix = np.zeros(Ne)
        for c1,c2 in G.edges:
            id_edge = G[c1][c2]['id_edge']
            edge_matrix[id_edge] = G[c1][c2]['length']
        c[d*Nc:] = s * edge_matrix

        print(c)

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
            c1,c2 = sorted((c1,c2)) #c1 < c2
            #Need the normal and the jump to be consistent. They're not now
            GG1[id_edge,2*c1] = n[0] #x component
            GG1[id_edge,2*c1+1] = n[1] #y component
            GG1[id_edge,2*c2] = -n[0] #x component
            GG1[id_edge,2*c2+1] = -n[1] #y component

        #Problems in the following.
        #Now writing the constraints for the absolute values
        GG2 = np.zeros((2*Ne, d*Nc+Ne))
        for c1,c2 in G.edges:
            id_edge = G[c1][c2]['id_edge']
            #print(c1,c2,id_edge)
            t = G[c1][c2]['tangent']
            #First inequality
            GG2[2*id_edge,2*c1] = -t[0] #x component
            GG2[2*id_edge,2*c1+1] = -t[1] #y component
            GG2[2*id_edge,2*c2] = t[0] #x component
            GG2[2*id_edge,2*c2+1] = t[1] #y component
            GG2[2*id_edge, d*Nc+id_edge] = -1 #for abs
            #Second inequality
            GG2[2*id_edge+1,2*c1] = t[0] #x component
            GG2[2*id_edge+1,2*c1+1] = t[1] #y component
            GG2[2*id_edge+1,2*c2] = -t[0] #x component
            GG2[2*id_edge+1,2*c2+1] = -t[1] #y component
            GG2[2*id_edge+1, d*Nc+id_edge] = -1 #for abs

        #Assembling constraints
        GG = np.concatenate((GG1, GG2))
        print(GG)
        self.G = matrix(GG, tc='d')
        
        #Right-hand side
        h = np.zeros(3*Ne)
        self.h = matrix(h, tc='d')
        

    def equality_constraint(self, GM): #Zero-average displacement
        G = GM.graph
        d = GM.d #dimension
        Nc = GM.Nc #Number of cells
        Ne = GM.Ne #Number of internal edges
        
        #rhs of the equality constraint
        b = np.zeros(d)
        self.b = matrix(b, tc='d')

        #lhs equality constraint
        A = np.arange(1, d*Nc+1)
        A %= 2
        B = (A + 1) % 2
        A = np.array((A, B))
        B = np.zeros((d,Ne))
        A = np.concatenate((A, B), axis=1)
        self.A = matrix(A, tc='d')

    def solve(self, d, Nc):
        sol = solvers.lp(self.E, self.G, self.h, self.A, self.b)

        #Checking we find no displacement
        assert sol['status'] == 'optimal'
        vec_sol = sol['x'] #result should be zero
        disp = np.array(vec_sol)[:d*Nc].reshape((Nc, d))
        assert np.linalg.norm(disp) < 1e-10
        #Check on y? Is it the vector sum of all disp?
        #print(sol['x'])
        #print(sol['y'])
        #print(sol['z'])

        #Test
        #print(self.G.size)
        #print(sol['z'].size)
        vec_forces = -self.G.T * sol['z'] #THIS IS IT?
        #print(aux[:d*Nc])

        #Returning forces in each cell
        #vec_forces = sol['z']
        return np.array(vec_forces)[:d*Nc].reshape((Nc, d))
        
