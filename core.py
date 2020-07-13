import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
import copy

def solve(q_start, q_end, n_wp):
    n_dof = len(q_start)
    q_start, q_end = np.array(q_start), np.array(q_end)
    step = (q_end - q_start)/(n_wp - 1)
    init_solution = np.hstack([q_start + i * step for i in range(n_wp)])
    return init_solution.reshape(n_wp * n_dof, 1)

def construct_Abc(q_start, q_end, n_wp): # A, b and c terms of chomp 
    n_dof = len(q_start)
    w_s = 10.0
    w_e = 10.0

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
        e = -q_start*w_s
        for i in range(n_wp - 2):
            e = np.hstack((e, np.zeros(n_dof)))
        e = np.hstack((e, -q_end*w_e))
        return e.reshape(n_wp*n_dof, 1)

    def construct_c():
        return w_s * np.linalg.norm(q_start) * 2 + w_e * np.linalg.norm(q_end) * 2

    A = construct_A()
    b = construct_b()
    c = construct_c()

    return A, b, c

def additional_cost(xi, n_dof, n_wp):
    Q = xi.reshape(n_wp, n_dof) 
    w = 0
    c = 10.0
    p = np.array([0.001, 0.0])
    for q in Q:
        diff = np.linalg.norm(q - p) - 0.2
        if diff < 0.0:
            w += c * diff*2
    return w

def compute_grad(f, x0, n_dim):
    f0 = f(x0)
    grad = np.zeros(n_dim)
    for i in range(n_dim):
        x_ = copy.copy(x0)
        dx = 1e-8
        x_[i] += dx 
        grad[i] = (f(x_) - f0)/dx
    return grad


n_wp = 20
n_dof = 2
s = np.array([-0.5]*n_dof)
e = np.array([0.5]*n_dof)
A, b, c = construct_Abc(s, e, n_wp)
Ainv = np.linalg.inv(A)

def cost_function(x, grad):
    f = lambda xi: (0.5 * xi.T.dot(A).dot(xi) + xi.T.dot(b)).item() 
    if grad.size > 0:
        g = compute_grad(f, x, len(x))
        for i in range(len(x)):
            grad[i] = g[i]
    return f(x)

def ineq_const(x, grad):
    def f(x):
        X = x.reshape(20, 2)
        val = 0.0
        for x in X:
            tmp = (x[0] ** 2 + x[1] ** 2  - 0.3**2)
            if tmp < 0:
                val -= tmp
        return val

    if grad.size > 0:
        g = compute_grad(f, x, len(x))
        for i in range(len(x)):
            grad[i] = g[i]
    return f(x)

xi = solve(s, e, n_wp)
xi += np.random.randn(40, 1) * 0.1


import nlopt

ndim = 40
algorithm = nlopt.LD_AUGLAG
opt = nlopt.opt(algorithm, ndim)
tol = 1e-8 
opt.set_ftol_rel(tol)
opt.set_min_objective(cost_function)
opt.add_inequality_constraint(ineq_const, 1e-5)
xopt = opt.optimize(xi.flatten())

xi = xopt.reshape(20, 2)
Q = xi.reshape(n_wp, n_dof)
plt.scatter(Q[:, 0], Q[:, 1])
plt.show()

"""
for i in range(1000):
    if i%10==11:
        Q = xi.reshape(n_wp, n_dof)
        plt.scatter(Q[:, 0], Q[:, 1])
        plt.show()
    g_add = compute_grad(f, xi, n_wp*n_dof)
    g = grad(xi) + g_add.reshape(n_wp*n_dof, 1)
    xi = xi - Ainv.dot(g) * 0.1
    print(np.linalg.norm(g))

Q = xi.reshape(n_wp, n_dof)
plt.scatter(Q[:, 0], Q[:, 1])
plt.show()
"""
