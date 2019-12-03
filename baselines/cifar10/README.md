# ResNet-20 on CIFAR-10

| Method | Train/Test NLL | Train/Test Accuracy | Train Runtime (min) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic | 0.019 / 0.310 | 99.5% / 92.1% | 45 (1 P100 GPU) | 274K |
| Dropout | 0.137 / 0.324 | 95.1% / 90.0% | 51 (1 P100 GPU) | 274K |
| Ensemble (size=5) | 0.011 / 0.184 | 99.9% / 94.1% | 45 (5 P100 GPU) | 1.37M |
| Variational inference | 0.136 / 0.382 | 95.5% / 89.1% | 75 (1 P100 GPU) | 420K |

We note results in the literature below. Note there are differences in the setup
(sometimes major), so take any comparisons with a grain of salt.

| Source | Method | Train/Test NLL | Train/Test Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| [`junyuseu/pytorch-cifar-models`](https://github.com/junyuseu/pytorch-cifar-models) | Deterministic | - | - / 91.67% | - | 270K |
| [`keras-team/keras`](https://keras.io/examples/cifar10_resnet) | Deterministic | - | - / 92.16% | 1.94 (1 1080Ti GPU) | 270K |
| [`kuangliu/pytorch-cifar`](https://github.com/kuangliu/pytorch-cifar) | Deterministic (ResNet-18) | - | - / 93.02% | - | 11.7M |
| [He et al. (2015)](https://arxiv.org/abs/1512.03385)<sup>1</sup> | Deterministic | - | - / 91.25% | - | 270K |
| | Deterministic (ResNet-32) | - | - / 92.49% | - | 460K |
| | Deterministic (ResNet-44) | - | - / 92.83% | - | 660K |
| | Deterministic (ResNet-56) | - | - / 93.03% | - | 850K |
| | Deterministic (ResNet-110) | - | - / 93.39% | - | 1.7M |
| [Louizos et al. (2017)](https://arxiv.org/abs/1705.08665)<sup>2</sup> | Group-normal Jeffreys | - | - / 91.2% | - | 998K |
| | Group-Horseshoe | - | - / 91.0% | - | 820K |
| [Molchanov et al. (2017)](https://arxiv.org/abs/1701.05369)<sup>2</sup> | Variational dropout | - | - / 92.7% | - | 304K |
| [Louizos et al. (2018)](https://arxiv.org/abs/1712.01312)<sup>3</sup> | L0 regularization | - | - / 96.17% | 200 epochs | - |
| [Anonymous (2019)](https://openreview.net/forum?id=rkglZyHtvH)<sup>4</sup> | Refined VI (no batchnorm) | - / 0.696 | - / 75.5% | 5.5 (1 P100 GPU) | - |
| | Refined VI (batchnorm) | - / 0.593 | - / 79.7% | 5.5 (1 P100 GPU) | - |
| | Refined VI hybrid (no batchnorm) | - / 0.432 | - / 85.8% | 4.5 (1 P100 GPU) | - |
| | Refined VI hybrid (batchnorm) | - / 0.423 | - / 85.6% | 4.5 (1 P100 GPU) | - |
| [Anonymous (2019)](https://openreview.net/forum?id=Sklf1yrYDr)<sup>5</sup> | Deterministic | - | - / 95.31% | 250 epochs | 7.43M |
| | BatchEnsemble | - | - / 95.94% | 375 epochs | 7.47M |
| | Ensemble | - | - / 96.30% | 250 epochs each | 29.7M |
| | Monte Carlo Dropout | - | - / 95.72% | 375 epochs | 7.43M |
| [Heek and Kalchbrenner (2019)](https://arxiv.org/abs/1908.03491)<sup>6</sup> | Deterministic | - / 0.243 | - / 94.4% | 1000 epochs (1 V100 GPU) | 850K |
| | Adaptive Thermostat Monte Carlo (single sample) | - / 0.303 | - / 92.4% | 1000 epochs (1 V100 GPU) | - |
| | Adaptive Thermostat Monte Carlo (multi-sample) | - / 0.194 | - / 93.9% | 1000 epochs (1 V100 GPU) | - |
| | Sampler-based Nose-Hoover Thermostat (single sample) | - / 0.343 | - / 91.7% | 1000 epochs (1 V100 GPU) | - |
| | Sampler-based Nose-Hoover Thermostat (multi-sample) | - / 0.211 | - / 93.5% | 1000 epochs (1 V100 GPU) | - |
| [Osawa et al. (2019)](https://arxiv.org/abs/1906.02506)<sup>7</sup>  | Variational Online Gauss-Newton | - / 0.48 | 91.6% / 84.3% | 2.38 (128 P100 GPUs) | - |
| [Ovadia et al. (2019)](https://arxiv.org/abs/1906.02530)<sup>8</sup> | Deterministic | - / 1.120 | - / 91% | - | 274K |
| | Dropout | - / 0.771 | - / 91% | - | 274K |
| | Ensemble | - / 0.653 | - | - / 93.5% | - |
| | Variational inference | - / 0.823 | 88% | - | 630K |

1. Trains on 45k examples.
2. Not a ResNet (VGG). Parameter count is guestimated from counting number of parameters in [original model](http://torch.ch/blog/2015/07/30/cifar.html) to be 14.9M multiplied by the compression rate.
3. Uses Wide ResNet (WRN-28-10).
4. Does not use data augmentation.
5. Uses ResNet-32 with 4x number of typical filters. Ensembles uses 4 members.
6. Uses ResNet-56 and modifies architecture. Cyclical learning rate.
7. Scales KL by an additional factor of 10.
8. Trains on 40k examples. Performs variational inference over only first convolutional layer of every residual block and final output layer. Has free parameter on normal prior's location. Uses scale hyperprior (and with a fixed scale parameter). NLL results are medians, not means; accuracies are guestimated from Figure 2's plot.

TODO(trandustin): Add column for Test runtime.

TODO(trandustin): Add column for Checkpoints.
