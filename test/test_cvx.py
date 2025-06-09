from cvxopt import matrix, solvers

#original
c = matrix([-4., -5.])
G = matrix([[1., 0.], [0., 1.]])
h = matrix([0., 0.])

##test
#c = matrix([-4., -5., -1,-2])
#G = matrix([[-1,0,0,-1], [0,-1,0,-1], [0,0,-1,0], [0,0,-1,0,]], tc='d')
#h = matrix([-1,-1,0,-1], tc='d')


#solving
sol = solvers.lp(c, G, h)

print(sol['x']) #primal solution
#print(sol['s'])
#print(dict(sol))
#print(sol['z']) #dual solution
