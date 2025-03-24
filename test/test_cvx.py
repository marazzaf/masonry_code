from cvxopt import matrix, solvers
import sys

# Define the coefficients of the objective function
c = matrix([-4.0, -5.0])

# Define the coefficients of the inequality constraints
G = matrix([[2.0, 1.0], [1.0, 2.0]], tc='d')
h = matrix([3.0, 3.0])

# Define the coefficients of the equality constraints
A = matrix([[1.0], [1.0]], tc='d')
b = matrix([4.0])

# Solve the linear optimization problem
solution = solvers.lp(c, G, h, A, b)

# Extract the optimal solution
print(solution['x'])

n=10
G = matrix(0.0, (n,n), 'd')
print(G)
G[::n+1] = -1.0
h = matrix(0.0, (n,1))
A = matrix(1.0, (1,n))
b = matrix(1.0)
