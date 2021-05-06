# Rank-1 BNNs

A Rank-1 Bayesian neural net (Rank-1 BNN) [1] is an efficient and scalable
approach to variational BNNs that posits prior distributions on rank-1 factors
of the weights and optimizes global mixture variational posterior distributions.

This directory contains the original scripts used for the paper, as well as
additional experimental setups.

NOTE: An updated version of the codebase, further expanding on the original
results, can be found in
[Uncertainty Baselines](https://github.com/google/uncertainty-baselines).

## References

> M. W. Dusenberry, G. Jerfel, Y. Wen, Y. Ma, J. Snoek, K. Heller, B. Lakshminarayanan, and D. Tran.
> [Efficient and scalable Bayesian neural nets with rank-1 factors.](https://arxiv.org/abs/2005.07186)
> In International Conference onMachine Learning, 2020.

```none
@inproceedings{dusenberry2020efficient,
  title={Efficient and Scalable {Bayesian} Neural Nets with Rank-1 Factors},
  author={Dusenberry, Michael W. and Jerfel, Ghassen and Wen, Yeming and Ma, Yian and Snoek, Jasper and Heller,
    Katherine and Lakshminarayanan, Balaji and Tran, Dustin},
  booktitle={International Conference on Machine Learning},
  year={2020},
}
```
