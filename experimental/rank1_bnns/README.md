# Rank-1 BNNs

A Rank-1 Bayesian neural net (Rank-1 BNN) [1] is an efficient and scalable
approach to variational BNNs that posits prior distributions on rank-1 factors
of the weights and optimizes global mixture variational posterior distributions.

This directory contains the original scripts used for the paper, as well as
additional experimental setups.

NOTE: An updated version of the Rank-1 BNNs codebase can be found in
[Uncertainty Baselines](https://github.com/google/uncertainty-baselines).

References:

  [1]: Michael W. Dusenberry\*, Ghassen Jerfel\*, Yeming Wen, Yian Ma, Jasper
       Snoek, Katherine Heller, Balaji Lakshminarayanan, Dustin Tran. Efficient
       and Scalable Bayesian Neural Nets with Rank-1 Factors. In Proc. of
       International Conference on Machine Learning (ICML) 2020.
       https://arxiv.org/abs/2005.07186
