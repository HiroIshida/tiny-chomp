import numpy as np
from functools import lru_cache

def solve(q_start, q_end, n_wp):
    n_dof = len(q_start)
    q_start, q_end = np.array(q_start), np.array(q_end)
    step = (q_end - q_start)/(n_wp - 1)
    init_solution = np.array([q_start + i * step for i in range(n_wp)])
    return init_solution

@lru_cache(None)
def construct_K(n_wp, n_dof):
    K_ = np.diag([1]*n_wp)
    for i in range(n_wp):
        for j in range(n_wp):
            if i>j and abs(i - j) == 1:
                K_[i, j] = -1
    K = np.kron(K_, np.eye(n_dof))
    return K

def construct_Abc(q_start, q_end, n_wp): # A, b and c terms of chomp 
    n_dof = len(q_start)
    K = construct_K(n_wp, n_dof)
    def construct_e():
        e = -q_start
        for i in range(n_wp - 2):
            e = np.hstack((e, np.zeros(n_dof)))
        e = np.hstack((e, q_end))
        return e.reshape(n_wp*n_dof, 1)

    e = construct_e()
    A = K.T.dot(K)
    b = K.T.dot(e)
    c = np.linalg.norm(e)**2
    return A, b, c

s = np.array([0, 0, 0])
e = np.array([1, 1, 1])
A, b, c = construct_Abc(s, e, 3)

