import numpy as np
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import scipy.io
from shaded_plt import shaded_plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import LogLocator, ScalarFormatter


def data2plt(data, color, marker, mark_every, name):
    data = np.array(data)

    # Calculate the mean, minimum, and maximum across each experiment run (axis=1 means along rows)
    means = np.mean(data, axis=0)
    # min_values = np.min(data, axis=0)
    # max_values = np.max(data, axis=0)
    std = np.std(data, axis=0)

    # X values representing the different runs (e.g., Run 1, Run 2, etc.)
    x = np.arange(0, data.shape[1])

    # Plot the mean line
    plt.plot(x, means, label=name, color=color, marker=marker, markevery=mark_every)

    # Fill the area between the min and max values for each experiment run
    plt.fill_between(x, means - std, means + std, color=color, alpha=0.2)

def read_plt(graph_type, n_node, nt, step_tpye, iid, dataset):
    pre_cd = 'res/' + dataset + '/'
    file_name = pre_cd + graph_type + str(n_node) + '_' + str(nt) + step_tpye + '_' + str(iid) + '_whole.mat'
    mat = scipy.io.loadmat(file_name)
    print('Loaded!')
    a = mat['sgd']
    a1 = mat['sgdm'][0,:,:]
    means = np.mean(a1, axis=0)
    std = np.std(a1, axis=0)
    T = a1.shape[1]

    mark_every = int(T * 0.1)
    font = FontProperties()
    font.set_size(18)
    font2 = FontProperties()
    font2.set_size(10)
    fig, ax = plt.subplots()
    # main plot
    data2plt(mat['sgdm'][0, :, :], 'b', '<', mark_every, 'CSGDM')
    data2plt(mat['sgd'][0, :, :], 'r', '<', mark_every, 'CSGD')
    data2plt(mat['dsgd'][0, :, :], 'y', 'd', mark_every, 'DSGD')
    data2plt(mat['edas'][0, :, :], 'k', 'v', mark_every, 'EDAS')
    data2plt(mat['dsgt'][0, :, :], 'c', 'H', mark_every, 'DSGT')
    data2plt(mat['dsmt'][0, :, :], 'g', 's', mark_every, 'DSMT')
    data2plt(mat['dsgt_hb'][0, :, :], 'm', 'h', mark_every, 'DSGT-HB')
    data2plt(mat['dsmt_noLCA'][0, :, :], 'indigo', '^', mark_every, 'DSMT_noLCA')
    # plt.grid(True)
    plt.yscale('log')

    # Use ScalarFormatter to apply scientific notation to the x-axis
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  # Controls when to use scientific notation

    plt.gca().xaxis.set_major_formatter(formatter)

    plt.tick_params(labelsize='large', width=3)
    plt.xlabel('# Iterations', fontproperties=font)
    plt.ylabel(r'$\mathbb{E}\|\nabla f(\bar{x}_k)\|^2$', fontsize=12)
    plt.legend(prop=font2, loc='lower left')

    # zoomed in plot
    # ring 100
    zoom_in_area = [7000, 10100, 1e-7, 6e-7]

    # ring 50
    # zoom_in_area = [7000, 10100, 2e-7, 10e-7]
    # Create an inset_axes object (a smaller axes in the main plot)
    axins = inset_axes(ax, width="55%", height="30%", loc='upper right')

    # Plot the zoomed-in data in the inset axes
    plt.sca(axins)
    data2plt(mat['sgdm'][0, :, :], 'b', '<', mark_every, 'CSGDM')
    data2plt(mat['sgd'][0, :, :], 'r', '<', mark_every, 'CSGD')
    data2plt(mat['dsgd'][0, :, :], 'y', 'd', mark_every, 'DSGD')
    data2plt(mat['edas'][0, :, :], 'k', 'v', mark_every, 'EDAS')
    data2plt(mat['dsgt'][0, :, :], 'c', 'H', mark_every, 'DSGT')
    data2plt(mat['dsmt'][0, :, :], 'g', 's', mark_every, 'DSMT')
    data2plt(mat['dsgt_hb'][0, :, :], 'm', 'h', mark_every, 'DSGT-HB')
    data2plt(mat['dsmt_noLCA'][0, :, :], 'indigo', '^', mark_every, 'DSMT_noLCA')

    plt.yscale('log')
    # Use ScalarFormatter to apply scientific notation to the x-axis
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  # Controls when to use scientific notation

    plt.gca().xaxis.set_major_formatter(formatter)

    # plt.grid(True, which='minor', color='grey', linestyle=':', linewidth=1)
    plt.grid(True, linestyle=':', linewidth=1)
    # Set limits for the zoomed-in area
    axins.set_xlim(zoom_in_area[0], zoom_in_area[1])
    axins.set_ylim(zoom_in_area[2], zoom_in_area[3])

    # Mark the zoomed-in area on the main plot
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", lw=2)

    plt.savefig('res/' + dataset + '/figs/' + graph_type + str(n_node) + '_' + str(nt) + step_tpye
                + '_' + str(iid) + '_shaded_std.pdf',
                format='pdf', dpi=4000, bbox_inches='tight')
    plt.show()

    # mark_every = int(T * 0.1)
    # font = FontProperties()
    # font.set_size(18)
    # font2 = FontProperties()
    # font2.set_size(10)
    # plt.figure()
    # shaded_plt(mat['sgdm'][0,:,:], 'b', '<', mark_every, 'CSGDM')
    # shaded_plt(mat['sgd'][0,:,:], 'r', '<', mark_every, 'CSGD')
    # shaded_plt(mat['dsgd'][0,:,:], 'y', 'd', mark_every, 'DSGD')
    # shaded_plt(mat['edas'][0,:,:], 'k', 'v', mark_every, 'EDAS')
    # shaded_plt(mat['dsgt'][0,:,:], 'c', 'H', mark_every, 'DSGT')
    # shaded_plt(mat['dsmt'][0,:,:], 'g', 's', mark_every, 'DSMT')
    # shaded_plt(mat['dsgt_hb'][0,:,:], 'm', 'h', mark_every, 'DSGT-HB')
    # shaded_plt(mat['dsmt_noLCA'][0,:,:], 'indigo', '^', mark_every, 'DSMT_noLCA')
    # plt.grid(True)
    # plt.yscale('log')
    # plt.tick_params(labelsize='large', width=3)
    # plt.xlabel('# Iterations', fontproperties=font)
    # plt.ylabel(r'$\mathbb{E}\|\nabla f(\bar{x}_k)\|^2$', fontsize=12)
    # plt.legend(prop=font2)
    # # plt.savefig('res/' + dataset + '/figs/' + graph_type + str(n_node) + '_' + str(nt) + step_tpye
    # #             + '_' + str(iid) + '_shaded_std.pdf',
    # #             format='pdf', dpi=4000, bbox_inches='tight')
    # plt.show()




if __name__ == '__main__':
    read_plt('ring', 100, 10, 'constant', False, 'cifar10')