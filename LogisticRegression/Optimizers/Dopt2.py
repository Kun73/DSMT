################################################################################################################################
##---------------------------------------------------Decentralized Optimizers-------------------------------------------------##
################################################################################################################################

import numpy as np
import copy as cp
import utilities as ut
from numpy import linalg as LA


def DSGD(prd, W, para, T, X_0, lca=False):
    # get stepsize
    if para['step'] == 'decreasing':
        def lr(k):
            return para['lr'][0] / (k + para['lr'][1])
    elif para['step'] == 'decay':
        def lr(k):
            if k <= int(T / 2):
                return para['lr'][0]
            elif k <= int(T * 0.8):
                return para['lr'][1]
            else:
                return para['lr'][2]
    else:
        def lr(k):
            return para['lr']

    inter = int(T / 100)

    # other parameters
    n_node = prd.n
    p = X_0.shape[1]
    # initialization
    X: object = cp.deepcopy(X_0)
    Xs = [cp.deepcopy(X)]
    I = np.eye(n_node)
    Wbar = (I + W) / 2
    if lca:
        # construct tW
        eigs = LA.eigvals(Wbar)
        eigs.sort()
        eig = eigs[n_node - 2]
        eta_root = (1 - np.sqrt(1 - eig ** 2)) / (1 + np.sqrt(1 - eig ** 2))
        eta_w = (1 + eta_root) / 2
        tWu = np.hstack(((1 + eta_w) * Wbar, -eta_w * I))
        tWl = np.hstack((I, np.zeros((n_node, n_node))))
        tW = np.vstack((tWu, tWl))
    if lca:
        tZ = np.vstack((X, X))
        for it in range(T):
            step = lr(it)
            sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
            grad = prd.networkgrad(X, sample_vec)
            tgrad = np.vstack((grad, grad))
            tZ = np.matmul(tW, tZ - step * tgrad)
            X = tZ[0:n_node, :]
            Xs.append(cp.deepcopy(X))
            ut.monitor('DSGD_LCA', it, T)
    else:
        for it in range(T):
            step = lr(it)
            sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
            grad = prd.networkgrad(X, sample_vec)
            X = np.matmul(Wbar, X - step * grad)
            Xs.append(cp.deepcopy(X))
            ut.monitor('DSGD', it, T)
    return Xs


def CSGD(prd, para, T, x_0):
    # get stepsize
    if para['step'] == 'decreasing':
        def lr(k):
            return para['lr'][0] / (k + para['lr'][1])
    elif para['step'] == 'decay':
        def lr(k):
            if k <= int(T / 2):
                return para['lr'][0]
            elif k <= int(T * 0.8):
                return para['lr'][1]
            else:
                return para['lr'][2]
    else:
        def lr(k):
            return para['lr']

    inter = int(T / 100)

    # other parameters
    n_node = prd.n
    p = x_0.shape[0]
    # initialization
    x: object = cp.deepcopy(x_0)
    xs = [cp.deepcopy(x)]

    for it in range(T):
        step = lr(it)
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
        x_re = np.tile(x, (n_node, 1))
        grad = prd.networkgrad(x_re, sample_vec)
        grad_a = np.mean(grad, axis=0)
        x = x - step * grad_a
        # if np.mod(it, inter) == 0:
        xs.append(cp.deepcopy(x))
        ut.monitor('SGD', it, T)
    return xs


def CSGDM(prd, beta, para, T, x_0):
    # get stepsize
    if para['step'] == 'decreasing':
        def lr(k):
            return para['lr'][0] / (k + para['lr'][1])
    elif para['step'] == 'decay':
        def lr(k):
            if k <= int(T / 2):
                return para['lr'][0]
            elif k <= int(T * 0.8):
                return para['lr'][1]
            else:
                return para['lr'][2]
    else:
        def lr(k):
            return para['lr']

    # other parameters
    n_node = prd.n
    p = x_0.shape[0]
    # initialization
    x = cp.deepcopy(x_0)
    xs = [cp.deepcopy(x)]
    for it in range(T):
        step = lr(it)
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
        x_re = np.tile(x, (n_node, 1))
        grad = prd.networkgrad(x_re, sample_vec)
        grad_a = np.mean(grad, axis=0)
        if it == 0:
            m = grad_a
        else:
            m = beta * m + (1 - beta) * grad_a
        x = x - step * m
        # if np.mod(it, inter) == 0:
        xs.append(cp.deepcopy(x))
        ut.monitor('SGDM', it, T)
    return xs


def EDAS(prd, W, para, T, X_0, lca=False):
    # get stepsize
    if para['step'] == 'decreasing':
        def lr(k):
            return para['lr'][0] / (k + para['lr'][1])
    elif para['step'] == 'decay':
        def lr(k):
            if k <= int(T / 2):
                return para['lr'][0]
            elif k <= int(T * 0.8):
                return para['lr'][1]
            else:
                return para['lr'][2]
    else:
        def lr(k):
            return para['lr']

    # other parameters
    n_node = prd.n
    p = X_0.shape[1]
    # initialization
    X = cp.deepcopy(X_0)
    Xs = [cp.deepcopy(X)]
    I = np.eye(n_node)
    Wbar = (I + W) / 2
    Y = np.zeros(X.shape)
    if lca:
        # construct tW
        eigs = LA.eigvals(Wbar)
        eigs = sorted(eigs)
        eig = eigs[n_node - 2]
        eta_root = (1 - np.sqrt(1 - eig ** 2)) / (1 + np.sqrt(1 - eig ** 2))
        eta_w = (1 + eta_root) / 2
        # eta_w = eta_root
        tWu = np.hstack(((1 + eta_w) * Wbar, -eta_w * I))
        tWl = np.hstack((I, np.zeros((n_node, n_node))))
        tW = np.vstack((tWu, tWl))
        tX = np.vstack((X, X))
        tY = np.vstack((Y, Y))
        # S = np.matmul(np.eye(2 * n_node) - tW, tW)
        # alpha = 0.001
        # tau = 0.2
        # gamma = 4 * alpha / (4 - 4 * tau - 3 * alpha)
        # tZ = tX
        # tD = np.vstack((Y, Y))
        # tU = tX
        I2 = np.eye(2 * n_node)
        # S = np.matmul(I2 - tW, tW)
        # tpsi = np.vstack((X, X))
        beta = eta_w / 2
        for it in range(T):
            step = lr(it)
            sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
            grad = prd.networkgrad(X, sample_vec)
            tgrad = np.vstack((grad, grad))
            if it == 0:
                tX_old = tX
                tgrad_old = tgrad
                tX = np.matmul(tW, tX - step * tgrad)
            else:
                tX_temp = tX - step * tgrad - tX_old - lr(it - 1) * tgrad_old
                tX_new = (1 - beta) * tX + beta * np.matmul(tW, tX_temp)
                tX_old = tX
                tgrad_old = tgrad
                tX = tX_new
            X = tX[0:n_node, :]
            # Y = beta * Y + (1 - beta) * tY[0:n_node, :]
            Xs.append(cp.deepcopy(X))
            ut.monitor('EDAS_LCA', it, T)
    else:
        for it in range(T):
            step = lr(it)
            sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
            grad = prd.networkgrad(X, sample_vec)
            if it == 0:
                X_old = X
                grad_old = grad
                X = np.matmul(Wbar, X - step * grad)
            else:
                X_new = np.matmul(Wbar, 2 * X - X_old - step * grad + lr(it - 1) * grad_old)
                X_old = X
                grad_old = grad
                X = X_new
                # Y = X_new
                # X = np.tile(np.mean(Y, 0), (n_node, 1))
            Xs.append(cp.deepcopy(X))
            ut.monitor('EDAS', it, T)
    return Xs


def DSGT(prd, W, para, T, X_0, lca=False):
    # get stepsize
    if para['step'] == 'decreasing':
        def lr(k):
            return para['lr'][0] / (k + para['lr'][1])
    elif para['step'] == 'decay':
        def lr(k):
            if k <= int(T / 2):
                return para['lr'][0]
            elif k <= int(T * 0.8):
                return para['lr'][1]
            else:
                return para['lr'][2]
    else:
        def lr(k):
            return para['lr']

    inter = int(T / 100)

    # other parameters
    n_node = prd.n
    p = X_0.shape[1]
    # initialization
    X = cp.deepcopy(X_0)
    Xs = [cp.deepcopy(X)]
    I = np.eye(n_node)
    Wbar = (I + W) / 2
    if lca:
        # construct tW
        eigs = LA.eigvals(Wbar)
        eigs.sort()
        eig = eigs[n_node - 2]
        eta_root = (1 - np.sqrt(1 - eig ** 2)) / (1 + np.sqrt(1 - eig ** 2))
        eta_w = (1 + eta_root) / 2
        tWu = np.hstack(((1 + eta_w) * Wbar, -eta_w * I))
        tWl = np.hstack((I, np.zeros((n_node, n_node))))
        tW = np.vstack((tWu, tWl))
        tX = np.vstack((X, X))
        # beta = 1 - (1 - eig)**(1/2)
        beta = np.sqrt(eta_w)
        for it in range(T):
            step = lr(it)
            # beta = step
            if it == 0:
                sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
                grad = prd.networkgrad(X, sample_vec)
                Y = (1 - beta) * grad
                G = (1 - beta) * grad
                tY = np.vstack((Y, Y))
            tX = np.matmul(tW, tX - step * np.vstack((Y, Y)))
            sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
            X = tX[0:n_node, :]
            grad_new = prd.networkgrad(X, sample_vec)
            G_new = beta * G + (1 - beta) * grad_new
            tY = np.matmul(tW, tY + np.vstack((G_new, G_new)) - np.vstack((G, G)))
            Y = tY[0:n_node, :]
            G = G_new
            # Xmean = np.mean(X, 0)
            Xs.append(cp.deepcopy(X))
            ut.monitor('DSGT_LCA', it, T)
    else:
        for it in range(T):
            step = lr(it)
            if it == 0:
                sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
                grad = prd.networkgrad(X, sample_vec)
                Y = grad
            X = np.matmul(Wbar, X - step * Y)
            sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
            grad_new = prd.networkgrad(X, sample_vec)
            Y = np.matmul(Wbar, Y + grad_new - grad)
            grad = grad_new
            Xs.append(cp.deepcopy(X))
            ut.monitor('DSGT', it, T)
    return Xs


def DSGT_HB(prd, W, para, T, X_0, beta):
    # get stepsize
    if para['step'] == 'decreasing':
        def lr(k):
            return para['lr'][0] / (k + para['lr'][1])
    elif para['step'] == 'decay':
        def lr(k):
            if k <= int(T / 2):
                return para['lr'][0]
            elif k <= int(T * 0.8):
                return para['lr'][1]
            else:
                return para['lr'][2]
    else:
        def lr(k):
            return para['lr']

    # other parameters
    n_node = prd.n
    p = X_0.shape[1]
    # initialization
    X = cp.deepcopy(X_0)
    Xs = [cp.deepcopy(X)]
    I = np.eye(n_node)
    Wbar = (I + W) / 2
    for it in range(T):
        step = lr(it)
        if it == 0:
            sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
            grad = prd.networkgrad(X, sample_vec)
            Y = grad
            U = Y
        X = np.matmul(Wbar, X - step * U)
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
        grad_new = prd.networkgrad(X, sample_vec)
        Y = np.matmul(Wbar, Y + grad_new - grad)
        U = beta * U + (1 - beta) * Y
        grad = grad_new
        Xs.append(cp.deepcopy(X))
        ut.monitor('DSGT-HB', it, T)
    return Xs


def DSMT(prd, W, para, T, X_0, lca=None):
    # get stepsize
    if para['step'] == 'decreasing':
        def lr(k):
            return para['lr'][0] / (k + para['lr'][1])
    elif para['step'] == 'decay':
        def lr(k):
            if k <= int(T / 2):
                return para['lr'][0]
            elif k <= int(T * 0.8):
                return para['lr'][1]
            else:
                return para['lr'][2]
    else:
        def lr(k):
            return para['lr']

    inter = int(T / 100)

    # other parameters
    n_node = prd.n
    p = X_0.shape[1]
    # initialization
    X = cp.deepcopy(X_0)
    Xl = cp.deepcopy(X_0)
    Xs = [cp.deepcopy(X)]
    I = np.eye(n_node)
    Wbar = (I + W) / 2
    eigs = LA.eigvals(Wbar)
    eigs.sort()
    eig = eigs[n_node - 2]
    eta_root = (1 - np.sqrt(1 - eig ** 2)) / (1 + np.sqrt(1 - eig ** 2))
    eta_w = (1 + eta_root) / 2
    beta = np.sqrt(eta_w)
    # beta = 1 - (1 - np.sqrt(eta_w)) / n_node**(1/2)
    # beta = 1 - (1 - eig)**(1/2)
    for it in range(T):
        step = lr(it)
        # beta = step
        if it == 0:
            sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
            grad = prd.networkgrad(X, sample_vec)
            Y = (1 - beta) * grad
            Yl = (1 - beta) * grad
            Z = (1 - beta) * grad
        X_temp = X - step * Y
        Xl_temp = Xl - step * Y
        X = (1 + eta_w) * np.matmul(Wbar, X_temp) - eta_w * Xl_temp
        Xl = X_temp
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
        grad_new = prd.networkgrad(X, sample_vec)
        Z_new = beta * Z + (1 - beta) * grad_new
        Y_temp = Y + Z_new - Z
        Yl_temp = Yl + Z_new - Z
        Y = (1 + eta_w) * np.matmul(Wbar, Y_temp) - eta_w * Yl_temp
        Yl = Y_temp
        Z = Z_new
        # Xmean = np.mean(X, 0)
        Xs.append(cp.deepcopy(X))
        ut.monitor('DSMT', it, T)
    return Xs

def DSMT_noLCA(prd, W, para, T, X_0, beta):
    # get stepsize
    if para['step'] == 'decreasing':
        def lr(k):
            return para['lr'][0] / (k + para['lr'][1])
    elif para['step'] == 'decay':
        def lr(k):
            if k <= int(T / 2):
                return para['lr'][0]
            elif k <= int(T * 0.8):
                return para['lr'][1]
            else:
                return para['lr'][2]
    else:
        def lr(k):
            return para['lr']

    # other parameters
    n_node = prd.n
    p = X_0.shape[1]
    # initialization
    X = cp.deepcopy(X_0)
    Xs = [cp.deepcopy(X)]
    I = np.eye(n_node)
    # Wbar = (I + W) / 2
    Wbar = W
    # eigs = LA.eigvals(Wbar)
    # eigs.sort()
    # eig = eigs[n_node - 2]
    # eta_root = (1 - np.sqrt(1 - eig ** 2)) / (1 + np.sqrt(1 - eig ** 2))
    # eta_w = (1 + eta_root) / 2
    # beta = np.sqrt(eta_w)
    # beta = 1 - (1 - eig)**(1/2)
    for it in range(T):
        step = lr(it)
        # beta = step
        if it == 0:
            sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
            grad = prd.networkgrad(X, sample_vec)
            Y = (1 - beta) * grad
            Z = (1 - beta) * grad
        X_temp = X - step * Y
        X = np.matmul(Wbar, X_temp)
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
        grad_new = prd.networkgrad(X, sample_vec)
        Z_new = beta * Z + (1 - beta) * grad_new
        Y_temp = Y + Z_new - Z
        Y = np.matmul(Wbar, Y_temp)
        Z = Z_new
        # Xmean = np.mean(X, 0)
        Xs.append(cp.deepcopy(X))
        ut.monitor('DSMT_noLCA', it, T)
    return Xs


