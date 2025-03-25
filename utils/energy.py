from graph import GranularMaterial
from cvxopt import solvers,matrix,spmatrix

class Energy:
    def __init__(self, graph):
        self.E = self.energy(graph.d, graph.Ne)
    

    def energy(self, d, Ne):
        aux = np.zeros(d * Ne)
        return matrix(aux, tc='d')

    def inequality_constraint(self, Ne):
        #Write the inequality constraint for compression
        x = n.flatten()
        I = np.array((np.arange(Ne), np.arange(Ne))).T.flatten()
        J = np.arange(2*Ne)
        G = spmatrix(x, I, J, tc='d') #left-hand side
        h = np.zeros(Ne) #right-hand side
        
