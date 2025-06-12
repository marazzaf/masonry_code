from cvxopt import matrix, solvers

#We want to minimize |x-3|+2|x+1| in x

# Variables: [x, t1, t2]
# Objective: minimize t1 + 2 * t2
c = matrix([0.0, 1.0, 2.0])

# Inequality constraints Gx <= h
# Constraints:
#  t1 - x >= -3  →  -x + t1 + 0 <= 3
# -t1 - x >= -3  →  -x - t1 + 0 <= 3
#  t2 - x >= 1   →  -x + 0 + t2 <= -1
# -t2 - x >= -1  →  -x + 0 - t2 <= 1

G = matrix([[1, -1, 1, -1], [-1, -1, 0, 0], [0, 0, -1, -1]], tc='d')

h = matrix([3, -3, -1, 1], tc='d')

# Solve the problem
sol = solvers.lp(c, G, h)

# Extract solution
x_opt = sol['x'] #Expect -1
print(x_opt)
print(sol['z'])
