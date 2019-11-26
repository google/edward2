# ResNet-20 on CIFAR-10

| Method | Train/Test NLL | Train/Test Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic | 0.088 / 0.350 | 97.0% / 90.6% | 24 (1 P100 GPU) | 274K |
| Dropout | 0.137 / 0.324 | 95.1% / 90.0% | 51 (1 P100 GPU) | 274K |
| Variational inference | 0.174 / 0.435 | 93.8% / 87.5% | 10-12 (1 P100 GPU) | 420K |

We note results in the literature below. Note there are differences in the setup
(sometimes major), so take any comparisons with a grain of salt.

| Source | Method | Train/Test NLL | Train/Test Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| [`junyuseu/pytorch-cifar-models`](https://github.com/junyuseu/pytorch-cifar-models) | Deterministic | - | - / 91.67% | - | 270K |
| [`keras-team/keras`](https://keras.io/examples/cifar10_resnet/) | Deterministic | - | - / 92.16% | 1.94 (1 1080Ti GPU) | 270K |
| [`kuangliu/pytorch-cifar`](https://github.com/kuangliu/pytorch-cifar) | Deterministic | - | - / 93.02% | - | 270K |
| [He et al. (2015)](https://arxiv.org/abs/1512.03385)<sup>1</sup> | Deterministic | - | - / 91.25% | - | 270K |
| [Anonymous (2019)](https://openreview.net/forum?id=rkglZyHtvH)<sup>2</sup> | Refined VI (no batchnorm) | - / 0.696 | - / 75.5% | 5.5 (1 P100 GPU) | - |
| | Refined VI (batchnorm) | - / 0.593 | - / 79.7% | 5.5 (1 P100 GPU) | - |
| | Refined VI hybrid (no batchnorm) | - / 0.432 | - / 85.8% | 4.5 (1 P100 GPU) | - |
| | Refined VI hybrid (batchnorm) | - / 0.423 | - / 85.6% | 4.5 (1 P100 GPU) | - |
| [Heek and Kalchbrenner (2019)](https://arxiv.org/abs/1908.03491)<sup>3</sup> | Deterministic | - / 0.243 | - / 94.4% | 1000 epochs (1 V100 GPU) | - |
| | Adaptive Thermostat Monte Carlo (single sample) | - / 0.303 | - / 92.4% | 1000 epochs (1 V100 GPU) | - |
| | Adaptive Thermostat Monte Carlo (multi-sample) | - / 0.194 | - / 93.9% | 1000 epochs (1 V100 GPU) | - |
| | Sampler-based Nose-Hoover Thermostat (single sample) | - / 0.343 | - / 91.7% | 1000 epochs (1 V100 GPU) | - |
| | Sampler-based Nose-Hoover Thermostat (multi-sample) | - / 0.211 | - / 93.5% | 1000 epochs (1 V100 GPU) | - |
| [Louizos et al. (2019)](https://arxiv.org/abs/1906.08324)<sup>4</sup> | Functional Neural Process | - | - / 93.6% | - | - |
| [Osawa et al. (2019)](https://arxiv.org/abs/1906.02506)<sup>5</sup>  | Variational Online Gauss-Newton | - / 0.48 | 91.6% / 84.3% | 2.38 (128 P100 GPUs) | - |
| [Ovadia et al. (2019)](https://arxiv.org/abs/1906.02530)<sup>6</sup> | Deterministic | - / 0.350 | - / 90.6% | 24 (1 P100 GPU) | 274K |
| | Variational inference | - / 0.440 | - / 87.3% | 10-12 (1 P100 GPU) | 630K |

1. Trains on 45k examples.
2. Does not use data augmentation.
3. Uses ResNet-56 and modifies architecture. Cyclical learning rate.
4. Not a ResNet.
5. Scales KL by an additional factor of 10.
6. Trains on 40k examples. Performs variational inference over only first convolutional layer of every residual block and final output layer. Has free parameter on normal prior's location. Uses scale hyperprior (and with a fixed scale parameter).

TODO(trandustin): Add column for Test runtime.

TODO(trandustin): Add column for Checkpoints.
