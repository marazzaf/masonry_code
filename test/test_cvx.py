from cvxopt import matrix, solvers

c = matrix([-4., -5.])
G = matrix([[1., 0.], [0., 1.]])
h = matrix([0., 0.])
sol = solvers.lp(c, G, h)

#print(sol['x'])
#print(sol['s'])
print(dict(sol))
#print(sol['z'])
