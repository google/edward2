# ResNet-50 on ImageNet

| Method | Train/Test NLL | Train/Test Top-1 Accuracy | Train/Test Top-5 Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| BatchEnsemble<sup>1</sup> | - | - / 76.13% | - | - (32 TPUv2 cores) | - |

We note results in the literature below. Note there are differences in the setup
(sometimes major), so take any comparisons with a grain of salt.

| Source | Method | Train/Test NLL | Train/Test Top-1 Accuracy | Train/Test Top-5 Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| [`facebookarchive/fb.resnet.torch `](https://github.com/facebookarchive/fb.resnet.torch ) | Deterministic | - | - / 75.99% | - / 92.98% | - | 25.6M |
| [`KaimingHe/deep-residual-networks`](https://github.com/KaimingHe/deep-residual-networks) | Deterministic | - | - / 75.3% | - | - | 25.6M |
| [`keras-team/keras`](https://keras.io/applications/#resnet) | Deterministic | - | - / 74.9% | - / 92.1% | - | 25.6M |
| | Deterministic (ResNet-152v2) | - | - / 78.0% | - / 94.2% | - | 60.3M |
| [`tensorflow/tpu`](https://github.com/tensorflow/tpu/tree/master/models/official/resnet)<sup>2</sup> | Deterministic | - | - / 76% | - | 17 (8 TPUv2) | 25.6M |
| [Heek and Kalchbrenner (2019)](https://arxiv.org/abs/1908.03491)<sup>3</sup> | Adaptive Thermostat Monte Carlo (single sample) | - / 1.08 | - / 74.2% | - | 1000 epochs (8 TPUv3 cores) | - |
| | Adaptive Thermostat Monte Carlo (multi-sample) | - / 0.883 | - / 77.5% | - | 1000 epochs (8 TPUv3 cores) | - |
| | Sampler-based Nose-Hoover Thermostat (single sample) | - / 1.15 | - / 73.1% | - | 1000 epochs (8 TPUv3 cores) | - |
| | Sampler-based Nose-Hoover Thermostat (multi-sample) | - / 0.941 | - / 76.4% | - | 1000 epochs (8 TPUv3 cores) | - |
| [Osawa et al. (2019)](https://arxiv.org/abs/1906.02506)<sup>4</sup>  | Variational Online Gauss-Newton | - / 1.37 | 73.87% / 67.38% | | 1.90 (128 P100 GPUs) | - |
| [Ovadia et al. (2019)](https://arxiv.org/abs/1906.02530) | Deterministic | - | - | - | - | - |
| | Dropout | - | - | - | - | - |
| | Ensemble | - | - | - | - | - |

1. Each ensemble member achieves roughly 75.6 test top-1 accuracy.
2. See documentation for differences from original paper, e.g., preprocessing.
3. Modifies architecture. Cyclical learning rate.
4. Uses ResNet-18. Scales KL by an additional factor of 5.

TODO(trandustin): Add column for Test runtime.

TODO(trandustin): Add column for Checkpoints.
