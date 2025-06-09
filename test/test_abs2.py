from cvxopt import matrix, solvers

# Variables: [u_1, u_2, t]
# Objective: minimize t - external forces
c = matrix([1, -1, 1], tc='d')

# Inequality constraints Gx <= h
G = matrix([[-1, 1, -1], [1, -1, 1], [0, -1, -1]], tc='d')
h = matrix([0, 0, 0], tc='d')

# Equality constraints Ax = b
A = matrix([[1], [1], [0]], tc='d')
b = matrix([0], tc='d')

# Solve the problem
sol = solvers.lp(c, G, h, A, b)

# Extract solution
x_opt = sol['x'] #Expect 0
print("Optimal x:", x_opt)
print("Dual z:", sol['z'])
