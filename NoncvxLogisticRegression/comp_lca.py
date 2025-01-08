import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from graph import Weight_matrix, Geometric_graph, Exponential_graph, Ring_graph, Grid_graph, ER_graph, RingPlus_graph
from analysis import error
from Problems.logistic_regression import LR_L2
from Problems.log_reg_cifar import LR_L4
from Optimizers import COPTIMIZER as copt
from alg_run import edas_run, dsgd_run, dsgt_run, csgd_run, csgdm_run, dsmt_run, dsgt_hb_run, dsmt_noLCA_run
import random
import os
import collections
from scipy.io import savemat
from shaded_plt import shaded_plt

def com_run(dataset, graph_type, n, dstep_size, iid=True, cepoch=400, T=100, cstep_size=0.1, momentum=0.95):
    if dataset == 'mnist':
        if iid:
            lr_0 = LR_L2(n, limited_labels=False, balanced=True)
        else:
            lr_0 = LR_L2(n, limited_labels=True, balanced=True)
    elif dataset == 'cifar10':
        if iid:
            lr_0 = LR_L4(n, limited_labels=False, balanced=True)
        else:
            lr_0 = LR_L4(n, limited_labels=True, balanced=True)
    else:
        print('Please choose from mnist or cifar10.')

    p = lr_0.p  # dimension of the model
    L = lr_0.L  # L-smooth constant
    mu = L / lr_0.kappa
    cstep_size = cstep_size / L

    # dstep_size = {
    #     'step': dstep_size['step'],
    #     'lr': [dstep_size['lr'][0] / mu, dstep_size['lr'][1]],
    # }


    """
    Initializing variables
    """
    # scaled initial points
    theta_c0 = np.random.normal(0, 1, p) / np.sqrt(n) * 2
    theta_0 = np.tile(theta_c0, (n, 1))
    if graph_type == 'grid':
        UG = Grid_graph(n).undirected()
    if graph_type == 'ring':
        UG = Ring_graph(n).undirected()
    if graph_type == 'er':
        random.seed(123)
        UG = ER_graph(n).undirected()
    if graph_type == 'exponential':
        UG = Exponential_graph(n).undirected()
    if graph_type == 'ringplus':
        random.seed(123)
        UG = RingPlus_graph(n).undirected()
    B = Weight_matrix(UG).metroplis()
    eigs = LA.eigvals((B + np.eye(n)) / 2)
    eigs.sort()
    eig = eigs[n - 2]
    graph_gap = eig
    print('lambda = ', graph_gap)

    """
    Centralized solutions
    """
    # solve the optimal solution of Logistic regression
    opt_name = 'theta_opt' + '_' + dataset + '.txt'
    if os.path.exists(opt_name):
        theta_opt = np.loadtxt(opt_name)
        F_opt = 0
    else:
        _, theta_opt, F_opt = copt.CNGD(lr_0, cstep_size, momentum, cepoch, theta_c0)
        np.savetxt(opt_name, theta_opt)

    # beta
    eta_root = (1 - np.sqrt(1 - graph_gap ** 2)) / (1 + np.sqrt(1 - graph_gap ** 2))
    eta_w = (1 + eta_root) / 2
    rhow = np.sqrt(eta_w)
    beta = 1 - (1 / n)**(1 / 3) * (1 - rhow)

    dsmt_no = dsmt_noLCA_run(theta_opt, F_opt, lr_0, beta, B, dstep_size, int(T), theta_0)
    dsmt = dsmt_run(theta_opt, F_opt, lr_0, B, dstep_size, int(T), theta_0)
    sgdm = csgdm_run(theta_opt, F_opt, lr_0, beta, dstep_size, int(T), theta_c0)
    sgd = csgd_run(theta_opt, F_opt, lr_0, B, dstep_size, int(T), theta_c0)
    dsgd = dsgd_run(theta_opt, F_opt, lr_0, B, dstep_size, int(T), theta_0, False)
    edas = edas_run(theta_opt, F_opt, lr_0, B, dstep_size, int(T), theta_0, False)
    dsgt_hb = dsgt_hb_run(theta_opt, F_opt, lr_0, beta, B, dstep_size, int(T), theta_0)
    dsgt = dsgt_run(theta_opt, F_opt, lr_0, B, dstep_size, int(T), theta_0, False)

    return sgd, sgdm, dsgd, dsgt, edas, dsgt_hb, dsmt, dsmt_no


if __name__ == '__main__':

    nt = 10
    dataset = 'cifar10'
    graph_type = 'ring'
    n = 100
    iid = False
    T = int(8e3)

    sgd = []
    sgdm = []
    dsgd = []
    dsgt = []
    edas = []
    dsgt_hb = []
    dsmt = []
    dsmt_no = []

    dstep_size = {
        'step': 'constant',
        'lr': 1 / 50
    }
    for it in range(nt):
        print('Implementing', it + 1, 'trail.....')
        tsgd, tsgdm, tdsgd, tdsgt, tedas, tdsgt_hb, tdsmt, tdsmt_noLCA= com_run(dataset=dataset, graph_type=graph_type, dstep_size=dstep_size,
                                                              iid=iid, n=n, cepoch=500, T=T, cstep_size=1, momentum=0.9)

        sgd.append(tsgd)
        sgdm.append(tsgdm)
        dsgd.append(tdsgd)
        dsgt.append(tdsgt)
        edas.append(tedas)
        dsgt_hb.append(tdsgt_hb)
        dsmt.append(tdsmt)
        dsmt_no.append(tdsmt_noLCA)

    sgdm_mean = np.sum(sgdm, axis=0) / nt
    dsgd_mean = np.sum(dsgd, axis=0) / nt
    edas_mean = np.sum(edas, axis=0) / nt
    dsmt_mean = np.sum(dsmt, axis=0) / nt
    dsgt_hb_mean = np.sum(dsgt_hb, axis=0) / nt
    sgd_mean = np.sum(sgd, axis=0) / nt
    dsgt_mean = np.sum(dsgt, axis=0) / nt
    dsmt_no_mean = np.sum(dsmt_no, axis=0) / nt

    # mark_every = int(T * 0.1)
    # font = FontProperties()
    # font.set_size(18)
    # font2 = FontProperties()
    # font2.set_size(10)
    # plt.figure()
    # plt.plot(sgdm_mean, '-<b', markevery=mark_every)
    # plt.plot(sgd_mean, '-<r', markevery=mark_every)
    # plt.plot(dsgd_mean, '-dy', markevery=mark_every)
    # # plt.plot(dsgd_lca, '-^m', markevery=mark_every)
    # plt.plot(edas_mean, '-vk', markevery=mark_every)
    # # plt.plot(edas_lca, '-hm', markevery=mark_every)
    # plt.plot(dsgt_mean, '-Hc', markevery=mark_every)
    # plt.plot(dsmt_mean, '-sg', markevery=mark_every)
    # plt.plot(dsgt_hb_mean, '-hm', markevery=mark_every)
    # plt.plot(dsmt_no_mean, '-^', markevery=mark_every)
    # plt.grid(True)
    # plt.yscale('log')
    # plt.tick_params(labelsize='large', width=3)
    # plt.xlabel('# Iterations', fontproperties=font)
    # plt.ylabel(r'$\mathbb{E}\|\nabla f(\bar{x}_k)\|^2$', fontsize=12)
    # # plt.legend(('DSGD' + '(' + str(tau) + ')', 'EDAS'+ '(' + str(tau) + ')', 'DSGT'+ '(' + str(tau) + ')',
    # #             'DSGT', 'EDAS', 'DSGD',), prop=font2)
    # plt.legend(('CSGDM', 'CSGD', 'DSGD', 'EDAS', 'DSGT', 'DSMT', 'DSGT-HB', 'DSMT_noLCA'), prop=font2)
    # plt.savefig('res/' + dataset + '/figs/' + graph_type + str(n) + '_' + str(nt) + dstep_size['step']
    #             + '_' + str(iid) + '.pdf',
    #             format='pdf', dpi=4000, bbox_inches='tight')
    # plt.show()

    mark_every = int(T * 0.1)
    font = FontProperties()
    font.set_size(18)
    font2 = FontProperties()
    font2.set_size(10)
    plt.figure()
    shaded_plt(sgdm, 'b', '<', mark_every, 'CSGDM')
    shaded_plt(sgd, 'r', '<', mark_every, 'CSGD')
    shaded_plt(dsgd, 'y', 'd', mark_every, 'DSGD')
    shaded_plt(edas, 'k', 'v', mark_every, 'EDAS')
    shaded_plt(dsgt, 'c', 'H', mark_every, 'DSGT')
    shaded_plt(dsmt, 'g', 's', mark_every, 'DSMT')
    shaded_plt(dsgt_hb, 'm', 'h', mark_every, 'DSGT-HB')
    shaded_plt(dsmt_no, 'indigo', '^', mark_every, 'DSMT_noLCA')
    plt.grid(True)
    plt.yscale('log')
    plt.tick_params(labelsize='large', width=3)
    plt.xlabel('# Iterations', fontproperties=font)
    plt.ylabel(r'$\mathbb{E}\|\nabla f(\bar{x}_k)\|^2$', fontsize=12)
    plt.legend(prop=font2)
    plt.savefig('res/' + dataset + '/figs/' + graph_type + str(n) + '_' + str(nt) + dstep_size['step']
                + '_' + str(iid) + '_shaded.pdf',
                format='pdf', dpi=4000, bbox_inches='tight')
    plt.show()

    ### save
    res_dict = collections.defaultdict(list)
    res_dict['sgd'].append(sgd_mean)
    res_dict['sgdm'].append(sgdm_mean)
    res_dict['dsgd'].append(dsgd_mean)
    res_dict['edas'].append(edas_mean)
    res_dict['dsgt'].append(dsgt_mean)
    res_dict['dsgt_hb'].append(dsgt_hb_mean)
    res_dict['dsmt'].append(dsmt_mean)
    res_dict['dsmt_noLCA'].append(dsmt_no_mean)
    res_dict['stepsize'].append(dstep_size)
    file_name = 'res/' + dataset + '/' + graph_type + str(n) + '_' + str(nt) + dstep_size['step'] + '_' + str(iid)
    savemat(file_name + ".mat", res_dict)

    ### save whole
    res_whole_dict = collections.defaultdict(list)
    res_whole_dict['sgd'].append(sgd)
    res_whole_dict['sgdm'].append(sgdm)
    res_whole_dict['dsgd'].append(dsgd)
    res_whole_dict['edas'].append(edas)
    res_whole_dict['dsgt'].append(dsgt)
    res_whole_dict['dsgt_hb'].append(dsgt_hb)
    res_whole_dict['dsmt'].append(dsmt)
    res_whole_dict['dsmt_noLCA'].append(dsmt_no)
    res_whole_dict['stepsize'].append(dstep_size)
    file_name = 'res/' + dataset + '/' + graph_type + str(n) + '_' + str(nt) + dstep_size['step'] + '_' + str(iid)
    savemat(file_name + "_whole.mat", res_whole_dict)

    print("Results saved! :)")

    # # test
    # com_run(dataset='mnist', graph_type='ring', dstep_size=dstep_size, iid=False, n=16, nt=1, cepoch=500,
    #           T=5e3, cstep_size=1, momentum=0.9)
