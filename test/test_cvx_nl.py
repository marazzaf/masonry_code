from cvxopt import solvers, matrix, spdiag

def acent(A, b):
    m, n = A.size
    def F(x=None, z=None):
        if x is None: return 0, matrix(1.0, (n,1))
        if min(x) <= 0.0: return None
        f = sum(abs(x))
        Df = matrix([1.0 if xi > 0 else (-1.0 if xi < 0 else 0.0) for xi in x]).T
        if z is None: return f, Df
        H = spdiag(z[0] * x**-2)
        return f, Df, H
    return solvers.cp(F, A=A, b=b)['x']

A = matrix([[2., 1.], [1., 2.]])
b = matrix([3., 3.])

sol = acent(A, b)
print(sol)
