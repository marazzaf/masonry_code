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
            #First inequality. Direction of -t
            GG2[2*id_edge,2*c1] = -t[0] #x component
            GG2[2*id_edge,2*c1+1] = -t[1] #y component
            GG2[2*id_edge,2*c2] = t[0] #x component
            GG2[2*id_edge,2*c2+1] = t[1] #y component
            GG2[2*id_edge, d*Nc+id_edge] = -1 #for abs
            #Second inequality. Direction of t
            GG2[2*id_edge+1,2*c1] = t[0] #x component
            GG2[2*id_edge+1,2*c1+1] = t[1] #y component
            GG2[2*id_edge+1,2*c2] = -t[0] #x component
            GG2[2*id_edge+1,2*c2+1] = -t[1] #y component
            GG2[2*id_edge+1, d*Nc+id_edge] = -1 #for abs

        #Assembling constraints
        GG = np.concatenate((GG1, GG2))
        print(GG.shape)
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

        #Test
        #Blocking disp of first cell
        self.A = spmatrix(1, [0, 1], [0, 1], size=(d,d*Nc+Ne))
        ##Block disp of boundary cells on bottom left and right
        #I = []
        #J = []
        #x = np.ones(2*len(self.bnd))
        #for c in self.bnd:
        #    #Continue

        ##lhs equality constraint
        #A = np.arange(1, d*Nc+1)
        #A %= 2
        #B = (A + 1) % 2
        #A = np.array((A, B))
        #B = np.zeros((d,Ne))
        #A = np.concatenate((A, B), axis=1)
        #print(A)
        ##self.A = matrix(A, tc='d')

    def solve(self, GM):
        #Solving linear problem
        sol = solvers.lp(self.E, self.G, self.h, self.A, self.b)

        #Checking solution converges
        assert sol['status'] == 'optimal'

        #Assembling forces at the internal edges
        d = GM.d
        Ne = GM.Ne
        aux1 = np.array(sol['z'][:Ne]).reshape(Ne) #Multiply by each edge normal to have the normal component of the force at each edge
        aux2 = sol['z'][Ne:]
        aux2 = np.array(aux2).reshape((Ne,d))
        aux2 = -aux2[:,0] + aux2[:,1] #Summing the components in each direction along t
        aux = np.array([aux1, aux2]).T #Force at each edge in (n,t) coordinates
        
        #Returning forces in each internal cell
        vec_forces = aux.copy()
        G =  GM.graph
        for c1,c2 in G.edges:
            id_edge = G[c1][c2]['id_edge']
            n = G[c1][c2]['normal']
            t = G[c1][c2]['tangent']
            vec_forces[id_edge,:] = vec_forces[0] * n + vec_forces[1] * t
        return vec_forces #Forces in (e_1,e_2) basis

        ##Returning forces in each cell
        #vec_forces = sol['z']
        #print(vec_forces)
        #sys.exit()
        #return np.array(vec_forces)[:d*Nc].reshape((Nc, d))
        
