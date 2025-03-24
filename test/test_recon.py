import numpy as np

#Fixing seed
np.random.seed(seed=136985)

A = np.random.uniform(size=8)
A = A.reshape((2,2,2))

b = np.random.uniform(size=4)
b = b.reshape((2,2))

#Barycentre
x_c = np.array([.5, .5])

#Barycentre of edges
x_e_1 = np.array([0, .5])
x_e_2 = np.array([.5, 0])
x_e_3 = np.array([1, .5])
x_e_4 = np.array([.5, 1])

#Stresses at the edges
sigma_1 = np.dot(A, x_e_1) + b
sigma_2 = np.dot(A, x_e_2) + b
sigma_3 = np.dot(A, x_e_3) + b
sigma_4 = np.dot(A, x_e_4) + b

#Normals at edges
n_1 = np.array([-1, 0])
n_2 = np.array([0, -1])
n_3 = np.array([1, 0])
n_4 = np.array([0, 1])

#Normal stresses at edges
sigma_1_n = np.dot(sigma_1, n_1)
sigma_2_n = np.dot(sigma_2, n_2)
sigma_3_n = np.dot(sigma_3, n_3)
sigma_4_n = np.dot(sigma_4, n_4)

#Components of the stress reconstruction
comp_1 = np.outer(sigma_1_n, .5 * n_1)
comp_2 = np.outer(sigma_2_n, .5 * n_2)
comp_3 = np.outer(sigma_3_n, .5 * n_3)
comp_4 = np.outer(sigma_4_n, .5 * n_4)

sigma_rec = comp_1 + comp_2 + comp_3 + comp_4
print(sigma_rec)

sigma_test = np.dot(A, x_c) + b
print(sigma_test)
