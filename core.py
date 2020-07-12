import numpy as np
from functools import lru_cache

def solve(q_start, q_end, n_wp):
    n_dof = len(q_start)
    q_start, q_end = np.array(q_start), np.array(q_end)
    step = (q_end - q_start)/(n_wp - 1)
    init_solution = np.array([q_start + i * step for i in range(n_wp)])
    return init_solution

@lru_cache(None)
def create_diagonal_matrix(n_dof, n_wp):
    corner_cases = [0, n_wp-1]
    diag = [(1 if i in corner_cases else 2) 
            for i in range(n_wp)]
    mat_ = np.diag(diag)
    for i in range(n_wp):
        for j in range(n_wp):
            if abs(i - j) == 1:
                mat_[i, j] = -1
    mat = np.kron(mat_, np.eye(n_dof))
    return mat

mat = create_diagonal_matrix(2, 2)
    


