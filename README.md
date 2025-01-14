# DSMT

This repo includes codes of the paper [An accelerated distributed stochastic gradient method with momentum](https://arxiv.org/abs/2402.09714). We use some code from: https://github.com/qureshi-mi/S-ADDOPT.


# SCVX

Please run `LogisticRegression/comp_lca.py` to obtain the results and run `LogisticRegression/read_plt_whole.py` to obtain the following figures:


![SCVX, ring 50](https://github.com/Kun73/DSMT/blob/main/LogisticRegression/res/cifar10/figs/ring50_10constant_False_shaded_std.pdf)

![SCVX, ring 100](https://github.com/Kun73/DSMT/blob/main/LogisticRegression/res/cifar10/figs/ring100_10constant_False_shaded_std.pdf)

# NCVX

Please run `NoncvxLogisticRegression/comp_lca.py` to obtain the results and run `NoncvxLogisticRegression/read_plt_whole.py` to obtain the following figures:

![NCVX, ring 50](https://github.com/Kun73/DSMT/blob/main/NoncvxLogisticRegression/res/cifar10/figs/ring50_10constant_False_shaded_std.pdf)

![NCVX, ring 100](https://github.com/Kun73/DSMT/blob/main/NoncvxLogisticRegression/res/cifar10/figs/ring100_10constant_False_shaded_std.pdf)


# Note 

If you find this paper interesting, please consider citing 
```
@article{huang2024accelerated,
  title={An accelerated distributed stochastic gradient method with momentum},
  author={Huang, Kun and Pu, Shi and Nedi{\'c}, Angelia},
  journal={arXiv preprint arXiv:2402.09714},
  year={2024}
}
```
