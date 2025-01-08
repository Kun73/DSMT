from analysis import error
from Optimizers import Dopt2 as dopt


def edas_run(theta_opt, F_opt, lr_0, B, dstep_size, T, theta_0, lca=False):

    error_lr_0 = error(lr_0, theta_opt, F_opt)

    # edas
    theta = dopt.EDAS(lr_0, B, dstep_size, T, theta_0, lca=lca)
    res_x = error_lr_0.theta_gap_path(theta)

    return res_x


def dsgd_run(theta_opt, F_opt, lr_0, B, dstep_size, T, theta_0, lca=False):

    error_lr_0 = error(lr_0, theta_opt, F_opt)

    # dsgd
    theta = dopt.DSGD(lr_0, B, dstep_size, T, theta_0, lca=lca)
    res_x = error_lr_0.theta_gap_path(theta)

    return res_x


def dsgt_run(theta_opt, F_opt, lr_0, B, dstep_size, T, theta_0, lca=False):

    error_lr_0 = error(lr_0, theta_opt, F_opt)

    # dsgt
    theta = dopt.DSGT(lr_0, B, dstep_size, T, theta_0, lca=lca)
    res_x = error_lr_0.theta_gap_path(theta)

    return res_x




def csgd_run(theta_opt, F_opt, lr_0, B, dstep_size, T, theta_0, lca=False):

    error_lr_0 = error(lr_0, theta_opt, F_opt)

    # CSGD
    theta = dopt.CSGD(lr_0, dstep_size, T, theta_0)
    res_x = error_lr_0.ctheta_gap_path(theta)

    return res_x


def csgdm_run(theta_opt, F_opt, lr_0, beta, dstep_size, T, theta_0):

    error_lr_0 = error(lr_0, theta_opt, F_opt)

    # CSGD
    theta = dopt.CSGDM(lr_0, beta, dstep_size, T, theta_0)
    res_x = error_lr_0.ctheta_gap_path(theta)

    return res_x


def dsmt_run(theta_opt, F_opt, lr_0, B, dstep_size, T, theta_0):

    error_lr_0 = error(lr_0, theta_opt, F_opt)

    # dsgt
    theta = dopt.DSMT(lr_0, B, dstep_size, T, theta_0)
    res_x = error_lr_0.theta_gap_path(theta)

    return res_x

def dsmt_noLCA_run(theta_opt, F_opt, lr_0, beta, B, dstep_size, T, theta_0):

    error_lr_0 = error(lr_0, theta_opt, F_opt)

    # dsgt
    theta = dopt.DSMT_noLCA(lr_0, B, dstep_size, T, theta_0, beta)
    res_x = error_lr_0.theta_gap_path(theta)

    return res_x


def dsgt_hb_run(theta_opt, F_opt, lr_0, beta, B, dstep_size, T, theta_0):

    error_lr_0 = error(lr_0, theta_opt, F_opt)

    # dsgt
    theta = dopt.DSGT_HB(lr_0, B, dstep_size, T, theta_0, beta)
    res_x = error_lr_0.theta_gap_path(theta)

    return res_x