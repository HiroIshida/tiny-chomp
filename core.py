import numpy as np
from functools import lru_cache

def solve(q_start, q_end, n_wp):
    n_dof = len(q_start)
    q_start, q_end = np.array(q_start), np.array(q_end)
    step = (q_end - q_start)/(n_wp - 1)
    init_solution = np.hstack([q_start + i * step for i in range(n_wp)])
    return init_solution.reshape(n_wp * n_dof, 1)

def construct_Abc(q_start, q_end, n_wp): # A, b and c terms of chomp 
    n_dof = len(q_start)
    w_s = 1000.0
    w_e = 1000.0

    def construct_A():
        corner_cases = [0, n_wp-1]
        diag = [(1 if i in corner_cases else 2)
                for i in range(n_wp)]
        mat_ = np.diag(diag)
        for i in range(n_wp):
            for j in range(n_wp):
                if abs(i - j) == 1:
                    mat_[i, j] = -1

        # terminal constraints
        mat_[0, 0] += w_s 
        mat_[n_wp-1, n_wp-1] += w_e
        mat = np.kron(mat_, np.eye(n_dof))
        return mat

    def construct_b():
        e = -2*q_start*w_s
        for i in range(n_wp - 2):
            e = np.hstack((e, np.zeros(n_dof)))
        e = np.hstack((e, -2*q_end*w_e))
        return e.reshape(n_wp*n_dof, 1)

    def construct_c():
        return w_s * np.linalg.norm(q_start) * 2 + w_e * np.linalg.norm(q_end) * 2

    A = construct_A()
    b = construct_b()
    c = construct_c()

    return A, b, c

s = np.array([0])
e = np.array([1])
A, b, c = construct_Abc(s, e, n_wp)

grad = lambda xi: A.dot(xi) + b
cost = lambda xi: (0.5 * xi.T.dot(A).dot(xi) + xi.T.dot(b)).item()

xi = solve(s, e, n_wp)
for i in range(100):
    g = grad(xi)
    xi = xi - g * 0.001
    print(g)

