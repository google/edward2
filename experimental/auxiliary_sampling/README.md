# Refining the variational posterior through iterative optimization

__Abstract.__
Variational inference (VI) is a popular approach for approximate Bayesian inference that is particularly promising for highly parameterized models such as deep neural networks.  A key challenge of variational inference is to approximate the posterior over model parameters with a distribution that is simpler and tractable yet sufficiently expressive. In this work, we propose a method for training highly flexible variational distributions by starting with a coarse approximation and iteratively refining it. Each refinement step makes cheap, local adjustments and only requires optimization of simple variational families. We demonstrate theoretically that our method always improves a bound on the approximation (the Evidence Lower BOund) and observe this empirically across a variety of benchmark tasks.  In experiments, our method consistently outperforms recent variational inference methods for deep learning in terms of log-likelihood and the ELBO.  We see that the gains are further amplified on larger scale models, significantly outperforming standard VI and deep ensembles on residual networks on CIFAR10.


## Experiments

The training binary for all variational inference experiments is
`run_training.py`. See `deterministic_baseline/run_det_training.py` for ensemble
and deterministic neural network experiments.

