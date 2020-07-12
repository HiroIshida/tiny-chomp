import numpy as np
from functools import lru_cache

def solve(q_start, q_end, n_wp):
    n_dof = len(q_start)
    q_start, q_end = np.array(q_start), np.array(q_end)
    step = (q_end - q_start)/(n_wp - 1)
    init_solution = np.array([q_start + i * step for i in range(n_wp)])
    return init_solution

@lru_cache(None)
def create_K(n_wp, n_dof):
    K_ = np.diag([1]*n_wp)
    for i in range(n_wp):
        for j in range(n_wp):
            if i>j and abs(i - j) == 1:
                K_[i, j] = -1
    K = np.kron(K_, np.eye(n_dof))
    return K

mat = create_K(3, 2)
